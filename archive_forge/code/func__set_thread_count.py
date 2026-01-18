import json
from .. import CatBoostError
from ..eval.factor_utils import FactorUtils
from ..core import _NumpyAwareEncoder
def _set_thread_count(self, thread_count):
    if thread_count is not None and thread_count != -1:
        params = self._params
        params['thread_count'] = thread_count
        self.__set_params(params)