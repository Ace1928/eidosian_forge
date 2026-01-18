from pathlib import Path
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import graph as graph_module
from autokeras import pipeline
from autokeras import tuners
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils
from autokeras.utils import utils
def _has_y(self, dataset):
    """Remove y from the tf.data.Dataset if exists."""
    shapes = data_utils.dataset_shape(dataset)
    if len(shapes) <= 1:
        return False
    for shape in shapes:
        if isinstance(shape, tuple):
            return True
    return len(shapes) == 2 and len(self.inputs) == 1 and (len(self.outputs) == 1)