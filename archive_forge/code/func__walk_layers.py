from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.tensorflow_stub import dtypes
def _walk_layers(keras_layer):
    """Walks the nested keras layer configuration in preorder.

    Args:
      keras_layer: Keras configuration from model.to_json.

    Yields:
      A tuple of (name_scope, layer_config).
      name_scope: a string representing a scope name, similar to that of tf.name_scope.
      layer_config: a dict representing a Keras layer configuration.
    """
    yield ('', keras_layer)
    if keras_layer.get('config').get('layers'):
        name_scope = keras_layer.get('config').get('name')
        for layer in keras_layer.get('config').get('layers'):
            for sub_name_scope, sublayer in _walk_layers(layer):
                sub_name_scope = '%s/%s' % (name_scope, sub_name_scope) if sub_name_scope else name_scope
                yield (sub_name_scope, sublayer)