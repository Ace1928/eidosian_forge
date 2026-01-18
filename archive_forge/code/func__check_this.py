import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def _check_this(self, msg=None):
    """Extend Dict._check_this() to check PWL validity."""
    if self.pwl is None:
        self._free()
    if self.pel is None:
        self._free()
    super()._check_this(msg)
    self.pwl._check_this(msg)
    self.pel._check_this(msg)