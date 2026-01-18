import collections
import dataclasses
import threading
from typing import Optional, Sequence, Tuple
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
from tensorboard.plugins.distribution import compressor
from tensorboard.util import tb_logging
def _ProcessHistogram(self, tag, wall_time, step, histo):
    """Processes a proto histogram by adding it to accumulated state."""
    histo = self._ConvertHistogramProtoToPopo(histo)
    histo_ev = HistogramEvent(wall_time, step, histo)
    self.histograms.AddItem(tag, histo_ev)
    self.compressed_histograms.AddItem(tag, histo_ev, self._CompressHistogram)