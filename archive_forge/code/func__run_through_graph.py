import collections
import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.config import backend
from keras.src.ops.operation import Operation
from keras.src.utils.nest import pack_sequence_as
def _run_through_graph(self, inputs, operation_fn):
    """Execute the graph.

        At each node we compute outputs via
        `operation_fn(node.operation)(*args, **kwargs)`.
        """
    inputs = tree.flatten(inputs)
    tensor_dict = {}
    for x, y in zip(self.inputs, inputs):
        tensor_dict[id(x)] = y
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = nodes_by_depth[depth]
        for node in nodes:
            if not node.operation or node.is_input:
                continue
            if any((id(x) not in tensor_dict for x in node.input_tensors)):
                continue
            args, kwargs = node.arguments.fill_in(tensor_dict)
            outputs = operation_fn(node.operation)(*args, **kwargs)
            for x, y in zip(node.outputs, tree.flatten(outputs)):
                tensor_dict[id(x)] = y
    output_tensors = []
    for x in self.outputs:
        output_tensors.append(tensor_dict[id(x)])
    return pack_sequence_as(self._outputs_struct, output_tensors)