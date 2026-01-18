import collections
import dataclasses
import operator
import re
from typing import Optional
from google.protobuf import struct_pb2
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context as backend_context_lib
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import metrics
from tensorboard.plugins.hparams import plugin_data_pb2
def _specifies_include(col_params):
    """Determines whether any `ColParam` contains the `include_in_result` field.

    In the case where none of the col_params contains the field, we should assume
    that all fields should be included in the response.
    """
    return any((col_param.HasField('include_in_result') for col_param in col_params))