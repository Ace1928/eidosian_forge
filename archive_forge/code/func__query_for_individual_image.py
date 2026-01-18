import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def _query_for_individual_image(self, run, tag, sample, index):
    """Builds a URL for accessing the specified image.

        This should be kept in sync with _serve_image_metadata. Note that the URL is
        *not* guaranteed to always return the same image, since images may be
        unloaded from the reservoir as new images come in.

        Args:
          run: The name of the run.
          tag: The tag.
          sample: The relevant sample index, zero-indexed. See documentation
            on `_image_response_for_run` for more details.
          index: The index of the image. Negative values are OK.

        Returns:
          A string representation of a URL that will load the index-th sampled image
          in the given run with the given tag.
        """
    query_string = urllib.parse.urlencode({'run': run, 'tag': tag, 'sample': sample, 'index': index})
    return query_string