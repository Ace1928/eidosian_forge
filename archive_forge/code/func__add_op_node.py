import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
def _add_op_node(op, func, input_dict):
    """Converts an op to a function def node and add it to `func`."""
    func.node_def.extend([_get_node_def(op)])
    node_def = func.node_def[-1]
    for i in range(len(node_def.input)):
        if not node_def.input[i].startswith('^'):
            assert node_def.input[i] in input_dict, '%s missing from %s' % (node_def.input[i], input_dict.items())
            node_def.input[i] = input_dict[node_def.input[i]]
    if op.op_def is not None and op.op_def.is_stateful:
        func.signature.is_stateful = True