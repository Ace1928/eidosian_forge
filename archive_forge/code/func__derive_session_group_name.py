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
def _derive_session_group_name(trial_id, hparams):
    if trial_id is not None:
        if not isinstance(trial_id, str):
            raise TypeError('`trial_id` should be a `str`, but got: %r' % (trial_id,))
        return trial_id
    jparams = json.dumps(hparams, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(jparams.encode('utf-8')).hexdigest()