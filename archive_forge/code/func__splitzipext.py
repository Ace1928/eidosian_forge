import os
import io
from .._utils import set_module
def _splitzipext(self, filename):
    """Split zip extension from filename and return filename.

        Returns
        -------
        base, zip_ext : {tuple}

        """
    if self._iszip(filename):
        return os.path.splitext(filename)
    else:
        return (filename, None)