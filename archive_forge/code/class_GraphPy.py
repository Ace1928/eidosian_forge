from collections import OrderedDict
import contextlib
from typing import Dict, Any
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
import torch
from ._proto_graph import node_proto
class GraphPy:
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization with TensorBoard.

    GraphDef generation operates in two passes:

    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.

    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """

    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)

    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for node_output, outputSize in zip(node.outputs, node.outputstensor_size):
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output, node.inputs, node.scopeName, outputSize, op_type=node.kind, attributes=node.attributes)
        self.find_common_root()
        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = node.scopeName + '/' + input_node_id
        for key, node in self.nodes_io.items():
            if type(node) == NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[key] = node.input_or_output + '/' + node.debugName
            if hasattr(node, 'scope') and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = self.shallowest_scope_name + '/' + node.debugName
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [self.unique_name_to_scoped_name[node_input_id] for node_input_id in node.inputs]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[node.debugName]

    def to_proto(self):
        """Convert graph representation of GraphPy object to TensorBoard required format."""
        nodes = []
        for v in self.nodes_io.values():
            nodes.append(node_proto(v.debugName, input=v.inputs, outputsize=v.tensor_size, op=v.kind, attributes=v.attributes))
        return nodes