import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
@dataclass
class ScheduleContext:
    """
        Encapsulates the context for processing one scheduling request.

        This exposes functions to read and write the scheduling nodes, to prevent
        accidental modification of the internal state.
        """
    _cluster_config: ClusterConfig
    _nodes: List[SchedulingNode] = field(default_factory=list)
    _node_type_available: Dict[NodeType, int] = field(default_factory=dict)

    def __init__(self, nodes: List[SchedulingNode], node_type_available: Dict[NodeType, int], cluster_config: ClusterConfig):
        self._nodes = nodes
        self._node_type_available = node_type_available
        self._cluster_config = cluster_config

    @classmethod
    def from_schedule_request(cls, req: SchedulingRequest) -> 'ResourceDemandScheduler.ScheduleContext':
        """
            Create a schedule context from a schedule request.
            It will populate the context with the existing nodes and the available node
            types from the config.

            Args:
                req: The scheduling request. The caller should make sure the
                    request is valid.
            """
        nodes = []
        for node in req.current_nodes:
            nodes.append(SchedulingNode(node_type=node.ray_node_type_name, total_resources=dict(node.total_resources), available_resources=dict(node.available_resources), labels=dict(node.dynamic_labels), status=SchedulingNodeStatus.RUNNING))
        cluster_config = req.cluster_config
        for instance in req.current_instances:
            if not is_pending(instance):
                continue
            node_config = cluster_config.node_type_configs[instance.ray_node_type_name]
            nodes.append(SchedulingNode.from_node_config(node_config, status=SchedulingNodeStatus.PENDING))
        node_type_available = cls._compute_available_node_types(nodes, req.cluster_config)
        return cls(nodes=nodes, node_type_available=node_type_available, cluster_config=req.cluster_config)

    @staticmethod
    def _compute_available_node_types(nodes: List[SchedulingNode], cluster_config: ClusterConfig) -> Dict[NodeType, int]:
        """
            Compute the number of nodes by node types available for launching based on
            the max number of workers in the config.
            Args:
                nodes: The current existing nodes.
                cluster_config: The cluster instances config.
            Returns:
                A dict of node types and the number of nodes available for launching.
            """
        node_type_available: Dict[NodeType, int] = defaultdict(int)
        node_type_existing: Dict[NodeType, int] = defaultdict(int)
        for node in nodes:
            node_type_existing[node.node_type] += 1
        for node_type, node_type_config in cluster_config.node_type_configs.items():
            node_type_available[node_type] = node_type_config.max_workers - node_type_existing.get(node_type, 0)
        return node_type_available

    def get_nodes(self) -> List[SchedulingNode]:
        return copy.deepcopy(self._nodes)

    def get_cluster_shape(self) -> Dict[NodeType, int]:
        cluster_shape = defaultdict(int)
        for node in self._nodes:
            cluster_shape[node.node_type] += 1
        return cluster_shape

    def update(self, new_nodes: List[SchedulingNode]) -> None:
        """
            Update the context with the new nodes.
            """
        self._nodes = new_nodes
        self._node_type_available = self._compute_available_node_types(self._nodes, self._cluster_config)

    def get_cluster_config(self) -> ClusterConfig:
        return self._cluster_config

    def __str__(self) -> str:
        return 'ScheduleContext({} nodes, node_type_available={}): {}'.format(len(self._nodes), self._node_type_available, self._nodes)