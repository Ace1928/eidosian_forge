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
def _format_image_mapping(mapping):
    """Prepares an image mapping for client consumption.

    Args:
        mapping: the result of DataProvider's `list_blob_sequences`.

    Returns:
        A dict with the following fields:
            tagRunSampledInfo: the return type of `_get_tag_run_image_info`
            tagDescriptions: the return type of `_get_tag_description_info`
    """
    return {'tagDescriptions': _get_tag_to_description(mapping), 'tagRunSampledInfo': _get_tag_run_image_info(mapping)}