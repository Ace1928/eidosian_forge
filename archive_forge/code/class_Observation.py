import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
@dataclasses.dataclass(frozen=True)
class Observation:
    """Contains the data within each Event file that the inspector cares about.

    The inspector accumulates Observations as it processes events.

    Attributes:
      step: Global step of the event.
      wall_time: Timestamp of the event in seconds.
      tag: Tag name associated with the event.
    """
    step: int
    wall_time: float
    tag: str