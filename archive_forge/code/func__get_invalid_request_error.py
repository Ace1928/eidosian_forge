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
def _get_invalid_request_error(self, series_request):
    tag = series_request.get('tag')
    plugin = series_request.get('plugin')
    run = series_request.get('run')
    sample = series_request.get('sample')
    if not isinstance(tag, str):
        return 'Missing tag'
    if plugin != scalar_metadata.PLUGIN_NAME and plugin != histogram_metadata.PLUGIN_NAME and (plugin != image_metadata.PLUGIN_NAME):
        return 'Invalid plugin'
    if plugin in _SINGLE_RUN_PLUGINS and (not isinstance(run, str)):
        return 'Missing run'
    if plugin in _SAMPLED_PLUGINS and (not isinstance(sample, int)):
        return 'Missing sample'
    return None