import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def _get_long_path_name(path):
    """Dummy no-op."""
    return path