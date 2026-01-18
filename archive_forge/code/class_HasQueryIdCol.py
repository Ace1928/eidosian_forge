from typing import Dict
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
class HasQueryIdCol(Params):
    """
    Mixin for param qid_col: query id column name.
    """
    qid_col = Param(Params._dummy(), 'qid_col', 'query id column name', typeConverter=TypeConverters.toString)