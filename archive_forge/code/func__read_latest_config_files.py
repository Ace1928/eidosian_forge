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
def _read_latest_config_files(self, run_path_pairs):
    """Reads and returns the projector config files in every run
        directory."""
    configs = {}
    config_fpaths = {}
    for run_name, assets_dir in run_path_pairs:
        config = ProjectorConfig()
        config_fpath = os.path.join(assets_dir, metadata.PROJECTOR_FILENAME)
        if tf.io.gfile.exists(config_fpath):
            with tf.io.gfile.GFile(config_fpath, 'r') as f:
                file_content = f.read()
            text_format.Parse(file_content, config)
        has_tensor_files = False
        for embedding in config.embeddings:
            if embedding.tensor_path:
                if not embedding.tensor_name:
                    embedding.tensor_name = os.path.basename(embedding.tensor_path)
                has_tensor_files = True
                break
        if not config.model_checkpoint_path:
            logdir = _assets_dir_to_logdir(assets_dir)
            ckpt_path = _find_latest_checkpoint(logdir)
            if not ckpt_path and (not has_tensor_files):
                continue
            if ckpt_path:
                config.model_checkpoint_path = ckpt_path
        if config.model_checkpoint_path and _using_tf() and (not tf.io.gfile.glob(config.model_checkpoint_path + '*')):
            logger.warning('Checkpoint file "%s" not found', config.model_checkpoint_path)
            continue
        configs[run_name] = config
        config_fpaths[run_name] = config_fpath
    return (configs, config_fpaths)