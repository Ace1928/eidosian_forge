import os
import time
from typing import List, Optional, Union, TYPE_CHECKING
import torch
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from ._convert_np import make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
from ._onnx_graph import load_onnx_graph
from ._pytorch_graph import graph
from ._utils import figure_to_image
from .summary import (
def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
    """Add audio data to summary.

        Args:
            tag (str): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_audio')
    if self._check_caffe2_blob(snd_tensor):
        from caffe2.python import workspace
        snd_tensor = workspace.FetchBlob(snd_tensor)
    self._get_file_writer().add_summary(audio(tag, snd_tensor, sample_rate=sample_rate), global_step, walltime)