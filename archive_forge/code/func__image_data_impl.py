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
def _image_data_impl(self, ctx, blob_key):
    """Gets the image data for a blob key.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            blob_key: a string identifier for a DataProvider blob.

        Returns:
            A tuple containing:
              data: a raw bytestring of the requested image's contents.
              content_type: a string HTTP content type.
        """
    data = self._data_provider.read_blob(ctx, blob_key=blob_key)
    image_type = imghdr.what(None, data)
    content_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    return (data, content_type)