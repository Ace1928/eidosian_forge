import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access."""
    return os.path.isdir(path) and os.access(path, os.W_OK)