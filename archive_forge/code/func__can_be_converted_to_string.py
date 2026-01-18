import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _can_be_converted_to_string(value):
    if not _protobuf_value_type(value):
        return False
    return json_format_compat.is_serializable_value(value)