import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _serialize_keras_object(instance):
    """Serialize a Keras object into a JSON-compatible representation."""
    _, instance = tf_decorator.unwrap(instance)
    if instance is None:
        return None
    if hasattr(instance, 'get_config'):
        name = instance.__class__.__name__
        config = instance.get_config()
        serialization_config = {}
        for key, item in config.items():
            if isinstance(item, six.string_types):
                serialization_config[key] = item
                continue
            try:
                serialized_item = _serialize_keras_object(item)
                if isinstance(serialized_item, dict) and (not isinstance(item, dict)):
                    serialized_item['__passive_serialization__'] = True
                serialization_config[key] = serialized_item
            except ValueError:
                serialization_config[key] = item
        return {'class_name': name, 'config': serialization_config}
    if hasattr(instance, '__name__'):
        return instance.__name__
    raise ValueError('Cannot serialize', instance)