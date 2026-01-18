from typing import Dict
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
class HasArbitraryParamsDict(Params):
    """
    This is a Params based class that is extended by _SparkXGBParams
    and holds the variable to store the **kwargs parts of the XGBoost
    input.
    """
    arbitrary_params_dict: 'Param[Dict]' = Param(Params._dummy(), 'arbitrary_params_dict', 'arbitrary_params_dict This parameter holds all of the additional parameters which are not exposed as the the XGBoost Spark estimator params but can be recognized by underlying XGBoost library. It is stored as a dictionary.')