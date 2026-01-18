import datetime
import logging
import os
import shutil
import time
import numpy
import pygloo
import ray
from ray._private import ray_constants
from ray.util.collective.collective_group import gloo_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
def _check_cpu_tensors(tensors):
    """Check only have one tensor and located on CPU."""
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("'tensors' must be a nonempty list.")
    if len(tensors) != 1:
        raise RuntimeError('Gloo only accept one tensor in the tensor list. Got {} != 1.'.format(len(tensors)))
    d = gloo_util.get_tensor_device(tensors[0])
    if d != 'cpu':
        raise RuntimeError('Gloo only accept cpu tensor . Got {}.'.format(d))