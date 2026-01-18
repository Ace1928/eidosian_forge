import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
Helper to read scalar or tensor data from the multiplexer.

        Args:
          convert_event: Takes `plugin_event_accumulator.TensorEvent` to
            either `provider.ScalarDatum` or `provider.TensorDatum`.
          index: The result of `self._index(...)`.
          downsample: Non-negative `int`; how many samples to return per
            time series.

        Returns:
          A dict of dicts of values returned by `convert_event` calls,
          suitable to be returned from `read_scalars` or `read_tensors`.
        