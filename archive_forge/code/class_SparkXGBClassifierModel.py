import warnings
from typing import Any, List, Optional, Type, Union
import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRanker, XGBRegressor
from .core import (  # type: ignore
from .utils import get_class_name
class SparkXGBClassifierModel(_ClassificationModel):
    """
    The model returned by :func:`xgboost.spark.SparkXGBClassifier.fit`

    .. Note:: This API is experimental.
    """

    @classmethod
    def _xgb_cls(cls) -> Type[XGBClassifier]:
        return XGBClassifier