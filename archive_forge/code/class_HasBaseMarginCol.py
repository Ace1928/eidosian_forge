from typing import Dict
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
class HasBaseMarginCol(Params):
    """
    This is a Params based class that is extended by _SparkXGBParams
    and holds the variable to store the base margin column part of XGboost.
    """
    base_margin_col = Param(Params._dummy(), 'base_margin_col', 'This stores the name for the column of the base margin', typeConverter=TypeConverters.toString)