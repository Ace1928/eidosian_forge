import os
import io
from .._utils import set_module
def _iswritemode(self, mode):
    """Test if the given mode will open a file for writing."""
    _writemodes = ('w', '+')
    for c in mode:
        if c in _writemodes:
            return True
    return False