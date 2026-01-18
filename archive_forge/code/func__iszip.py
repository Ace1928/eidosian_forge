import os
import io
from .._utils import set_module
def _iszip(self, filename):
    """Test if the filename is a zip file by looking at the file extension.

        """
    fname, ext = os.path.splitext(filename)
    return ext in _file_openers.keys()