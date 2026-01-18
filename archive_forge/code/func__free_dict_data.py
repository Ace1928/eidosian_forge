import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def _free_dict_data(self, dict):
    """Free the underlying pointer for a dict."""
    self._check_this()
    _e.broker_free_dict(self._this, dict)
    self._live_dicts[dict] -= 1
    if self._live_dicts[dict] == 0:
        del self._live_dicts[dict]