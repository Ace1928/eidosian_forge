import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
@dataclasses.dataclass(frozen=True)
class _Measurement:
    """Holds a session's metric value.

    Attributes:
      metric_value: Metric value of the session.
      session_index: Index of the session in its group.
    """
    metric_value: Optional[api_pb2.MetricValue]
    session_index: int