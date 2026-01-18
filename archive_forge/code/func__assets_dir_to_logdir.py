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
def _assets_dir_to_logdir(assets_dir):
    sub_path = os.path.sep + metadata.PLUGINS_DIR + os.path.sep
    if sub_path in assets_dir:
        two_parents_up = os.pardir + os.path.sep + os.pardir
        return os.path.abspath(os.path.join(assets_dir, two_parents_up))
    return assets_dir