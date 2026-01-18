import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
def _get_tensor_metadata(self, event, content_type, components, data_shape, config):
    """Converts a TensorDatum into a JSON-compatible response.

        Args:
          event: TensorDatum object containing data in proto format.
          content_type: enum plugin_data_pb2.MeshPluginData.ContentType value,
            representing content type in TensorDatum.
          components: Bitmask representing all parts (vertices, colors, etc.) that
            belong to the summary.
          data_shape: list of dimensions sizes of the tensor.
          config: rendering scene configuration as dictionary.

        Returns:
          Dictionary of transformed metadata.
        """
    return {'wall_time': event.wall_time, 'step': event.step, 'content_type': content_type, 'components': components, 'config': config, 'data_shape': list(data_shape)}