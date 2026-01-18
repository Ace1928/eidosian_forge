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
def _get_file_writer(self):
    """Return the default FileWriter instance. Recreates it if closed."""
    if self.all_writers is None or self.file_writer is None:
        self.file_writer = FileWriter(self.log_dir, self.max_queue, self.flush_secs, self.filename_suffix)
        self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
        if self.purge_step is not None:
            most_recent_step = self.purge_step
            self.file_writer.add_event(Event(step=most_recent_step, file_version='brain.Event:2'))
            self.file_writer.add_event(Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START)))
            self.purge_step = None
    return self.file_writer