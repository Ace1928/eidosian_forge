import abc
from typing import Callable, Dict, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import (
from ray.data import Preprocessor
from ray.train import Checkpoint
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI(stability='beta')
class PredictorNotSerializableException(RuntimeError):
    """Error raised when trying to serialize a Predictor instance."""
    pass