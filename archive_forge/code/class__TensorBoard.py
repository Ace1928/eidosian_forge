from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (
class _TensorBoard(TensorBoard, metaclass=ExceptionSafeClass):
    pass