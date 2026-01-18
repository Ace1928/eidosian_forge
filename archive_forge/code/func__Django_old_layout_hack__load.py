import io
import os
import pickle
import sys
import runpy
import types
import warnings
from . import get_start_method, set_start_method
from . import process
from . import util
def _Django_old_layout_hack__load():
    try:
        sys.path.append(os.environ['DJANGO_PROJECT_DIR'])
    except KeyError:
        pass