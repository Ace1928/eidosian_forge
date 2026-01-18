import abc
from typing import Callable, Dict, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.air.data_batch_type import DataBatchType
from ray.air.util.data_batch_conversion import (
from ray.data import Preprocessor
from ray.train import Checkpoint
from ray.util.annotations import DeveloperAPI, PublicAPI
class PandasUDFPredictor(Predictor):

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, **kwargs) -> 'Predictor':
        return PandasUDFPredictor()

    def _predict_pandas(self, df, **kwargs) -> 'pd.DataFrame':
        return pandas_udf(df, **kwargs)