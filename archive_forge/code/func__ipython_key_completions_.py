from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def _ipython_key_completions_(self):
    """ Custom tab completions for __getitem__ in IPython >=5.0. """
    return sorted(self.keys())