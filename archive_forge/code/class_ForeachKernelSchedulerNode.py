import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """Scheduler node which consists of a list of scheduler nodes that each operate on a
    distinct tensor in a list of tensors."""

    def get_consumer_subnode_for(self, producer):
        if producer.get_name() in self.read_to_node:
            return self.read_to_node[producer.get_name()]
        return None

    def get_producer_subnode_for(self, consumer):
        for rd in consumer.read_writes.reads:
            if rd.name in self.name_to_node:
                return self.name_to_node[rd.name]
        return None

    @classmethod
    def can_fuse(cls, producer, consumer):
        why = WhyNoFuse(producer, consumer)
        if producer.is_foreach() and consumer.is_foreach():
            foreach_match = len(producer.snodes) == len(consumer.snodes)
            if not foreach_match:
                why('foreach do not have same length')
            return foreach_match and all((producer.scheduler.can_fuse(l, r) for l, r in zip(producer.snodes, consumer.snodes)))
        elif consumer.is_foreach():
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)
            why('candidate producer is not dep of any foreach consumer')
            return False
        elif producer.is_foreach():
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)
            why('candidate consumer has no dep in any foreach producer')
            return False
        raise AssertionError('At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node')

    @classmethod
    def fuse(cls, producer, consumer):
        assert producer.is_foreach() or consumer.is_foreach()
        prev_node_1 = None
        prev_node_2 = None
        if producer.is_foreach() and consumer.is_foreach():
            fused_nodes = [FusedSchedulerNode.fuse(l, r) for l, r in zip(producer.snodes, consumer.snodes)]
        elif producer.is_foreach():
            producer_subnode = producer.get_producer_subnode_for(consumer)
            fused_nodes = []
            prev_node_1 = producer
            prev_node_2 = None
            for node in producer.snodes:
                if node is producer_subnode:
                    new_node = FusedSchedulerNode.fuse(node, consumer)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)
        elif consumer.is_foreach():
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            fused_nodes = []
            prev_node_1 = consumer
            prev_node_2 = None
            for node in consumer.snodes:
                if node is consumer_subnode:
                    new_node = FusedSchedulerNode.fuse(producer, node)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)
        return cls(producer.scheduler, fused_nodes, prev_node_1, prev_node_2)

    def __init__(self, scheduler: 'Scheduler', nodes: List[SchedulerNode], prev_node_1=None, prev_node_2=None):
        self.read_to_node = {}
        self.name_to_node = {}
        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, nodes)
            for node in nodes:
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node
                for name in node.get_names():
                    self.name_to_node[name] = node
        else:
            self.scheduler = scheduler
            self.snodes = nodes
            self.node: ir.Buffer = None
            self.users: List[NodeUser] = []
            self.set_read_writes(dependencies.ReadWrites.merge_list([prev_node_1.read_writes, prev_node_2.read_writes]))
            self.unmet_dependencies = {dep for dep in set.union(prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies) if dep.name not in self.get_names()} - self.read_writes.writes
            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])
            foreach_node = prev_node_1 if prev_node_1.is_foreach() else prev_node_2
            other_node = prev_node_2 if prev_node_1.is_foreach() else prev_node_1
            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)
            self.name_to_node = foreach_node.name_to_node
            for name in other_node.get_names():
                self.name_to_node[name] = other_node
        self.group = (nodes[0].get_device(), 'foreach')
        self.origins: Set[torch.fx.Node] = set()

    def mark_run(self):
        raise NotImplementedError

    def codegen(self):
        assert isinstance(self.node, ir.ComputedBuffer), f'type(self.node)={type(self.node)!r}'
        self.node.get_store_function()(self.node.make_loader()())

    def can_free(self):
        return NotImplementedError

    def is_foreach(self):
        return True

    def get_subkernel_nodes(self):
        """Returns a list of nodes which comprise the foreach kernel, operating on corresponding elements of our input lists.
        These nodes may be vertically fused."""
        return list(self.snodes)

    def get_nodes(self):
        """Returns all nodes contained in this kernel, unpacking fused nodes into their constituent scheduler nodes."""
        return list(itertools.chain(*[x.get_nodes() for x in self.snodes]))

    def get_first_name(self):
        return self.snodes[0].get_first_name()

    def prune_redundant_deps(self, name_to_fused_node):
        for node in self.snodes:
            node.prune_redundant_deps(name_to_fused_node)