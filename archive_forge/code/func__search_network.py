import keras_tuner
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks as blocks_module
from autokeras import keras_layers
from autokeras import nodes as nodes_module
from autokeras.engine import head as head_module
from autokeras.engine import serializable
from autokeras.utils import io_utils
def _search_network(self, input_node, outputs, in_stack_nodes, visited_nodes):
    visited_nodes.add(input_node)
    in_stack_nodes.add(input_node)
    outputs_reached = False
    if input_node in outputs:
        outputs_reached = True
    for block in input_node.out_blocks:
        for output_node in block.outputs:
            if output_node in in_stack_nodes:
                raise ValueError('The network has a cycle.')
            if output_node not in visited_nodes:
                self._search_network(output_node, outputs, in_stack_nodes, visited_nodes)
            if output_node in self._node_to_id.keys():
                outputs_reached = True
    if outputs_reached:
        self._add_node(input_node)
    in_stack_nodes.remove(input_node)