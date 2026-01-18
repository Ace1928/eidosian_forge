import json
from .. import CatBoostError
from ..eval.factor_utils import FactorUtils
from ..core import _NumpyAwareEncoder

            Instances of this class are cases which will be compared during evaluation
            Params are CatBoost params
            label is a string which will be used for plots and other visualisations
            ignored_features is a set of additional feature indices to ignore
        