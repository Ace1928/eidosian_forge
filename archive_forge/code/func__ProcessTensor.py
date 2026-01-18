import collections
import dataclasses
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import event_util
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.event_processing import reservoir
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import meta_graph_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tb_logging
def _ProcessTensor(self, tag, wall_time, step, tensor):
    tv = TensorEvent(wall_time=wall_time, step=step, tensor_proto=tensor)
    with self._tensors_by_tag_lock:
        if tag not in self.tensors_by_tag:
            reservoir_size = self._GetTensorReservoirSize(tag)
            self.tensors_by_tag[tag] = reservoir.Reservoir(reservoir_size)
    self.tensors_by_tag[tag].AddItem(_TENSOR_RESERVOIR_KEY, tv)