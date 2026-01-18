from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.tensorflow_stub import dtypes
def _update_dicts(name_scope, model_layer, input_to_in_layer, model_name_to_output, prev_node_name):
    """Updates input_to_in_layer, model_name_to_output, and prev_node_name
    based on the model_layer.

    Args:
      name_scope: a string representing a scope name, similar to that of tf.name_scope.
      model_layer: a dict representing a Keras model configuration.
      input_to_in_layer: a dict mapping Keras.layers.Input to inbound layer.
      model_name_to_output: a dict mapping Keras Model name to output layer of the model.
      prev_node_name: a string representing a previous, in sequential model layout,
                      node name.

    Returns:
      A tuple of (input_to_in_layer, model_name_to_output, prev_node_name).
      input_to_in_layer: a dict mapping Keras.layers.Input to inbound layer.
      model_name_to_output: a dict mapping Keras Model name to output layer of the model.
      prev_node_name: a string representing a previous, in sequential model layout,
                      node name.
    """
    layer_config = model_layer.get('config')
    if not layer_config.get('layers'):
        raise ValueError('layer is not a model.')
    node_name = _scoped_name(name_scope, layer_config.get('name'))
    input_layers = layer_config.get('input_layers')
    output_layers = layer_config.get('output_layers')
    inbound_nodes = model_layer.get('inbound_nodes')
    is_functional_model = bool(input_layers and output_layers)
    is_parent_functional_model = bool(inbound_nodes)
    if is_parent_functional_model and is_functional_model:
        for input_layer, inbound_node in zip(input_layers, inbound_nodes):
            input_layer_name = _scoped_name(node_name, input_layer)
            inbound_node_name = _scoped_name(name_scope, inbound_node[0])
            input_to_in_layer[input_layer_name] = inbound_node_name
    elif is_parent_functional_model and (not is_functional_model):
        prev_node_name = _scoped_name(name_scope, inbound_nodes[0][0][0])
    elif not is_parent_functional_model and prev_node_name and is_functional_model:
        assert len(input_layers) == 1, 'Cannot have multi-input Functional model when parent model is not Functional. Number of input layers: %d' % len(input_layer)
        input_layer = input_layers[0]
        input_layer_name = _scoped_name(node_name, input_layer)
        input_to_in_layer[input_layer_name] = prev_node_name
    if is_functional_model and output_layers:
        layers = _norm_to_list_of_layers(output_layers)
        layer_names = [_scoped_name(node_name, layer[0]) for layer in layers]
        model_name_to_output[node_name] = layer_names
    else:
        last_layer = layer_config.get('layers')[-1]
        last_layer_name = last_layer.get('config').get('name')
        output_node = _scoped_name(node_name, last_layer_name)
        model_name_to_output[node_name] = [output_node]
    return (input_to_in_layer, model_name_to_output, prev_node_name)