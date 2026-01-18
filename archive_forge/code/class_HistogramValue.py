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
@dataclasses.dataclass(frozen=True)
class HistogramValue:
    """Holds the information of the histogram values.

    Attributes:
      min: A float or int min value.
      max: A float or int max value.
      num: Total number of values.
      sum: Sum of all values.
      sum_squares: Sum of squares for all values.
      bucket_limit: Upper values per bucket.
      bucket: Numbers of values per bucket.
    """
    min: float
    max: float
    num: int
    sum: float
    sum_squares: float
    bucket_limit: Sequence[float]
    bucket: Sequence[int]