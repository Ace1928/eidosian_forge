import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def _get_generic_data_individual_image(self, ctx, blob_key):
    """Returns the actual image bytes for a given image.

        Args:
          blob_key: As returned by a previous `read_blob_sequences` call.

        Returns:
          A bytestring of the raw image bytes.
        """
    return self._data_provider.read_blob(ctx, blob_key=blob_key)