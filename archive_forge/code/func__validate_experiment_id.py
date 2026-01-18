import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _validate_experiment_id(self, experiment_id):
    if not isinstance(experiment_id, str):
        raise TypeError('experiment_id must be %r, but got %r: %r' % (str, type(experiment_id), experiment_id))