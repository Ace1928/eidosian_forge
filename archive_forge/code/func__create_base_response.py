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
def _create_base_response(self, series_request):
    tag = series_request.get('tag')
    run = series_request.get('run')
    plugin = series_request.get('plugin')
    sample = series_request.get('sample')
    response = {'plugin': plugin, 'tag': tag}
    if isinstance(run, str):
        response['run'] = run
    if isinstance(sample, int):
        response['sample'] = sample
    return response