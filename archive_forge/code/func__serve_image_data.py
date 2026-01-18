import collections
import imghdr
import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
@wrappers.Request.application
def _serve_image_data(self, request):
    """Serves an individual image."""
    ctx = plugin_util.context(request.environ)
    blob_key = request.args['imageId']
    if not blob_key:
        raise errors.InvalidArgumentError("Missing 'imageId' field")
    data, content_type = self._image_data_impl(ctx, blob_key)
    return http_util.Respond(request, data, content_type)