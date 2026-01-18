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
def _format_basic_mapping(mapping):
    """Prepares a scalar or histogram mapping for client consumption.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A dict with the following fields:
            runTagInfo: the return type of `_get_run_tag_info`
            tagDescriptions: the return type of `_get_tag_to_description`
    """
    return {'runTagInfo': _get_run_tag_info(mapping), 'tagDescriptions': _get_tag_to_description(mapping)}