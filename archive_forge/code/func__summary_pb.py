import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def _summary_pb(tag, hparams_plugin_data):
    """Create a summary holding the given `HParamsPluginData` message.

    Args:
      tag: The `str` tag to use.
      hparams_plugin_data: The `HParamsPluginData` message to use.

    Returns:
      A TensorBoard `summary_pb2.Summary` message.
    """
    summary = summary_pb2.Summary()
    summary_metadata = metadata.create_summary_metadata(hparams_plugin_data)
    value = summary.value.add(tag=tag, metadata=summary_metadata, tensor=metadata.NULL_TENSOR)
    return summary