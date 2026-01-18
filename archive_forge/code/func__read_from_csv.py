import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import pandas as pd
import tensorflow as tf
from tensorflow import nest
from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tuners import task_specific
from autokeras.utils import types
@staticmethod
def _read_from_csv(x, y):
    df = pd.read_csv(x)
    target = df.pop(y).to_numpy()
    return (df, target)