import os
import io
from .._utils import set_module
def _findfile(self, path):
    """Extend DataSource method to prepend baseurl to ``path``."""
    return DataSource._findfile(self, self._fullpath(path))