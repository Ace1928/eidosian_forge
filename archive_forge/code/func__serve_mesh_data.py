import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
@wrappers.Request.application
def _serve_mesh_data(self, request):
    """A route that returns data for particular summary of specified type.

        Data can represent vertices coordinates, vertices indices in faces,
        vertices colors and so on. Each mesh may have different combination of
        abovementioned data and each type/part of mesh summary must be served as
        separate roundtrip to the server.

        Args:
          request: werkzeug.Request containing content_type as a name of enum
            plugin_data_pb2.MeshPluginData.ContentType.

        Returns:
          werkzeug.Response either float32 or int32 data in binary format.
        """
    step = float(request.args.get('step', 0.0))
    tensor_events = self._collect_tensor_events(request, step)
    content_type = request.args.get('content_type')
    try:
        content_type = plugin_data_pb2.MeshPluginData.ContentType.Value(content_type)
    except ValueError:
        return http_util.Respond(request, 'Bad content_type', 'text/plain', 400)
    sample = int(request.args.get('sample', 0))
    response = [self._get_tensor_data(tensor, sample) for meta, tensor in tensor_events if meta.content_type == content_type]
    np_type = {plugin_data_pb2.MeshPluginData.VERTEX: np.float32, plugin_data_pb2.MeshPluginData.FACE: np.int32, plugin_data_pb2.MeshPluginData.COLOR: np.uint8}[content_type]
    response = np.array(response, dtype=np_type)
    response = response.reshape(-1).tobytes()
    return http_util.Respond(request, response, 'arraybuffer')