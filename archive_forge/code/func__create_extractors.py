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
def _create_extractors(col_params):
    """Creates extractors to extract properties corresponding to 'col_params'.

    Args:
      col_params: List of ListSessionGroupsRequest.ColParam protobufs.
    Returns:
      A list of extractor functions. The ith element in the
      returned list extracts the column corresponding to the ith element of
      _request.col_params
    """
    result = []
    for col_param in col_params:
        result.append(_create_extractor(col_param))
    return result