import json
from .. import CatBoostError
from ..eval.factor_utils import FactorUtils
from ..core import _NumpyAwareEncoder
def __set_params(self, params):
    self._params = params
    self._params_hash = hash(json.dumps(self._params, sort_keys=True, cls=_NumpyAwareEncoder))