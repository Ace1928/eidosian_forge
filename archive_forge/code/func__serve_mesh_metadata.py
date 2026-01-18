import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
@wrappers.Request.application
def _serve_mesh_metadata(self, request):
    """A route that returns the mesh metadata associated with a tag.

        Metadata consists of wall time, type of elements in tensor, scene
        configuration and so on.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A JSON list of mesh data associated with the run and tag
          combination.
        """
    tensor_events = self._collect_tensor_events(request)
    response = [self._get_tensor_metadata(tensor, meta.content_type, meta.components, meta.shape, meta.json_config) for meta, tensor in tensor_events]
    return http_util.Respond(request, response, 'application/json')