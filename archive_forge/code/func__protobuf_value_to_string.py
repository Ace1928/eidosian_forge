import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _protobuf_value_to_string(value):
    """Returns a string representation of given google.protobuf.Value message.

    Args:
      value: google.protobuf.Value message. Assumed to be of type 'number',
        'string' or 'bool'.
    """
    value_in_json = json_format.MessageToJson(value)
    if value.HasField('string_value'):
        return value_in_json[1:-1]
    return value_in_json