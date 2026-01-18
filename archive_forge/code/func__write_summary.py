import json
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard.util import tensor_util
def _write_summary(name, description, tensor, content_type, components, json_config, step):
    """Creates a tensor summary with summary metadata.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.
      tensor: Tensor to display in summary.
      content_type: Type of content inside the Tensor.
      components: Bitmask representing present parts (vertices, colors, etc.) that
        belong to the summary.
      json_config: A string, JSON-serialized dictionary of ThreeJS classes
        configuration.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.

    Returns:
      A boolean indicating if summary was saved successfully or not.
    """
    tensor = tf.convert_to_tensor(value=tensor)
    shape = tensor.shape.as_list()
    shape = [dim if dim is not None else -1 for dim in shape]
    tensor_metadata = metadata.create_summary_metadata(name, None, content_type, components, shape, description, json_config=json_config)
    return tf.summary.write(tag=metadata.get_instance_name(name, content_type), tensor=tensor, step=step, metadata=tensor_metadata)