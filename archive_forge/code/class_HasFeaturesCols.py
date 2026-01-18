from typing import Dict
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
class HasFeaturesCols(Params):
    """
    Mixin for param features_cols: a list of feature column names.
    This parameter is taken effect only when use_gpu is enabled.
    """
    features_cols = Param(Params._dummy(), 'features_cols', 'feature column names.', typeConverter=TypeConverters.toListString)

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(features_cols=[])