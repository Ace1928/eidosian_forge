import collections
import functools
import imghdr
import mimetypes
import os
import threading
import numpy as np
from werkzeug import wrappers
from google.protobuf import json_format
from google.protobuf import text_format
from tensorboard import context
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.http_util import Respond
from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from tensorboard.plugins.projector import metadata
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.util import tb_logging
@wrappers.Request.application
def _serve_config(self, request):
    run = request.args.get('run')
    if run is None:
        return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
    self._update_configs()
    config = self._configs.get(run)
    if config is None:
        return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
    return Respond(request, json_format.MessageToJson(config), 'application/json')