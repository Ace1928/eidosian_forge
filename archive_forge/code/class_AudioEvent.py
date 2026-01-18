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
class AudioEvent:
    """Contains information of an audio event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      encoded_audio_string: Audio content encoded in bytes.
      content_type: A string describes the type of the audio content.
      sample_rate: Sample rate of the audio in Hz. Must be positive.
      length_frames: Length of the audio in frames (samples per channel).
    """
    wall_time: float
    step: int
    encoded_audio_string: bytes
    content_type: str
    sample_rate: float
    length_frames: int