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
def _create_response(self, session_groups):
    return api_pb2.ListSessionGroupsResponse(session_groups=session_groups[self._request.start_index:self._request.start_index + self._request.slice_size], total_size=len(session_groups))