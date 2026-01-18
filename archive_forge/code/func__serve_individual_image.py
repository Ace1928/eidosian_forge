import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
@wrappers.Request.application
def _serve_individual_image(self, request):
    """Serves an individual image."""
    try:
        ctx = plugin_util.context(request.environ)
        blob_key = request.args['blob_key']
        data = self._get_generic_data_individual_image(ctx, blob_key)
    except (KeyError, IndexError):
        return http_util.Respond(request, 'Invalid run, tag, index, or sample', 'text/plain', code=400)
    image_type = imghdr.what(None, data)
    content_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    return http_util.Respond(request, data, content_type)