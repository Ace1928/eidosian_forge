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
def add_figure(self, tag: str, figure: Union['Figure', List['Figure']], global_step: Optional[int]=None, close: bool=True, walltime: Optional[float]=None) -> None:
    """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag: Data identifier
            figure: Figure or a list of figures
            global_step: Global step value to record
            close: Flag to automatically close the figure
            walltime: Optional override default walltime (time.time())
              seconds after epoch of event
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_figure')
    if isinstance(figure, list):
        self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='NCHW')
    else:
        self.add_image(tag, figure_to_image(figure, close), global_step, walltime, dataformats='CHW')