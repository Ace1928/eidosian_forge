import abc
from typing import Callable, Dict, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import (
from ray.data import Preprocessor
from ray.train import Checkpoint
from ray.util.annotations import DeveloperAPI, PublicAPI
def _set_cast_tensor_columns(self):
    """Enable automatic tensor column casting.

        If this is called on a predictor, the predictor will cast tensor columns to
        NumPy ndarrays in the input to the preprocessors and cast tensor columns back to
        the tensor extension type in the prediction outputs.
        """
    self._cast_tensor_columns = True