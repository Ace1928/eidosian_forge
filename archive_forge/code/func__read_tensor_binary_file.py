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
def _read_tensor_binary_file(fpath, shape):
    if len(shape) != 2:
        raise ValueError('Tensor must be 2D, got shape {}'.format(shape))
    tensor = np.fromfile(fpath, dtype='float32')
    return tensor.reshape(shape)