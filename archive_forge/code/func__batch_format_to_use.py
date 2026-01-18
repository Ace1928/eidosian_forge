import abc
from typing import Callable, Dict, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import (
from ray.data import Preprocessor
from ray.train import Checkpoint
from ray.util.annotations import DeveloperAPI, PublicAPI
@classmethod
def _batch_format_to_use(cls) -> BatchFormat:
    """Determine the batch format to use for the predictor."""
    has_pandas_implemented = cls._predict_pandas != Predictor._predict_pandas
    has_numpy_implemented = cls._predict_numpy != Predictor._predict_numpy
    if has_pandas_implemented and has_numpy_implemented:
        return cls.preferred_batch_format()
    elif has_pandas_implemented:
        return BatchFormat.PANDAS
    elif has_numpy_implemented:
        return BatchFormat.NUMPY
    else:
        raise NotImplementedError(f'Predictor {cls.__name__} must implement at least one of `_predict_pandas` and `_predict_numpy`.')