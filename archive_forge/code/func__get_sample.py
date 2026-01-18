import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
def _get_sample(self, tensor_datum, sample):
    """Returns a single sample from a batch of samples."""
    return tensor_datum.numpy[sample].tolist()