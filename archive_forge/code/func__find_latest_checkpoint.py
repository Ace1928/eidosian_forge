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
def _find_latest_checkpoint(dir_path):
    if not _using_tf():
        return None
    try:
        ckpt_path = tf.train.latest_checkpoint(dir_path)
        if not ckpt_path:
            ckpt_path = tf.train.latest_checkpoint(os.path.join(dir_path, os.pardir))
        return ckpt_path
    except tf.errors.NotFoundError:
        return None