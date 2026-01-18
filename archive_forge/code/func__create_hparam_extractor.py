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
def _create_hparam_extractor(hparam_name):
    """Returns an extractor function that extracts an hparam from a session
    group.

    Args:
      hparam_name: str. Identies the hparam to extract from the session group.
    Returns:
      A function that takes a tensorboard.hparams.SessionGroup protobuffer and
      returns the value, as a native Python object, of the hparam identified by
      'hparam_name'.
    """

    def extractor_fn(session_group):
        if hparam_name in session_group.hparams:
            return _value_to_python(session_group.hparams[hparam_name])
        return None
    return extractor_fn