import keras_tuner
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks as blocks_module
from autokeras import keras_layers
from autokeras import nodes as nodes_module
from autokeras.engine import head as head_module
from autokeras.engine import serializable
from autokeras.utils import io_utils
def _build_network(self):
    self._node_to_id = {}
    for input_node in self.inputs:
        self._search_network(input_node, self.outputs, set(), set())
    self._nodes = sorted(list(self._node_to_id.keys()), key=lambda x: self._node_to_id[x])
    for node in self.inputs + self.outputs:
        if node not in self._node_to_id:
            raise ValueError('Inputs and outputs not connected.')
    blocks = []
    for input_node in self._nodes:
        for block in input_node.out_blocks:
            if any([output_node in self._node_to_id for output_node in block.outputs]) and block not in blocks:
                blocks.append(block)
    for block in blocks:
        for input_node in block.inputs:
            if input_node not in self._node_to_id:
                raise ValueError('A required input is missing for HyperModel {name}.'.format(name=block.name))
    in_degree = [0] * len(self._nodes)
    for node_id, node in enumerate(self._nodes):
        in_degree[node_id] = len([block for block in node.in_blocks if block in blocks])
    self.blocks = []
    self._block_to_id = {}
    while len(blocks) != 0:
        new_added = []
        for block in blocks:
            if any([in_degree[self._node_to_id[node]] for node in block.inputs]):
                continue
            new_added.append(block)
        for block in new_added:
            blocks.remove(block)
        for block in new_added:
            self._add_block(block)
            for output_node in block.outputs:
                output_node_id = self._node_to_id[output_node]
                in_degree[output_node_id] -= 1