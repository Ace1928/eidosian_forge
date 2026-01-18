from tensorboard.compat.proto import graph_pb2
def _add_with_prepended_names(prefix, graph_to_add, destination_graph):
    for node in graph_to_add.node:
        new_node = destination_graph.node.add()
        new_node.CopyFrom(node)
        new_node.name = _prefixed_op_name(prefix, node.name)
        new_node.input[:] = [_prefixed_op_name(prefix, input_name) for input_name in node.input]
        if new_node.op == 'PartitionedCall' and new_node.attr['f']:
            new_node.attr['f'].func.name = _prefixed_func_name(prefix, new_node.attr['f'].func.name)
    for func in graph_to_add.library.function:
        new_func = destination_graph.library.function.add()
        new_func.CopyFrom(func)
        new_func.signature.name = _prefixed_func_name(prefix, new_func.signature.name)
    for gradient in graph_to_add.library.gradient:
        new_gradient = destination_graph.library.gradient.add()
        new_gradient.CopyFrom(gradient)
        new_gradient.function_name = _prefixed_func_name(prefix, new_gradient.function_name)
        new_gradient.gradient_func = _prefixed_func_name(prefix, new_gradient.gradient_func)