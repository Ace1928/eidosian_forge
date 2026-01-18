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
def _create_extractor(col_param):
    if col_param.HasField('metric'):
        return _create_metric_extractor(col_param.metric)
    elif col_param.HasField('hparam'):
        return _create_hparam_extractor(col_param.hparam)
    else:
        raise error.HParamsError('Got ColParam with both "metric" and "hparam" fields unset: %s' % col_param)