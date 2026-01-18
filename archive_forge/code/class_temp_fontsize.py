from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
class temp_fontsize(object):
    """Context manager to temporarily change matplotlib font size."""

    def __init__(self, size=None):
        """Initialize the context manager."""
        if size is None:
            size = plt.rcParams['font.size']
        self.size = size

    def __enter__(self):
        """Temporarily set the font size."""
        self.old_size = plt.rcParams['font.size']
        plt.rcParams['font.size'] = self.size

    def __exit__(self, type, value, traceback):
        """Change the font size back to default."""
        plt.rcParams['font.size'] = self.old_size