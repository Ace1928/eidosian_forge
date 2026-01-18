import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
@contextmanager
def _use_zordered_offset(self):
    if self._offset_zordered is None:
        yield
    else:
        old_offset = self._offsets
        super().set_offsets(self._offset_zordered)
        try:
            yield
        finally:
            self._offsets = old_offset