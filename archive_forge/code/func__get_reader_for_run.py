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
def _get_reader_for_run(self, run):
    if run in self.readers:
        return self.readers[run]
    config = self._configs[run]
    reader = None
    if config.model_checkpoint_path and _using_tf():
        try:
            reader = tf.train.load_checkpoint(config.model_checkpoint_path)
        except Exception:
            logger.warning('Failed reading "%s"', config.model_checkpoint_path)
    self.readers[run] = reader
    return reader