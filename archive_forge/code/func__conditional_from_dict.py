from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _conditional_from_dict(self, cont, by_name, names):
    if isinstance(cont, dict):
        if not by_name:
            raise ValueError('not by name, yet a dictionary was passed.')
        cont, tp = self._array_from_dict(cont, names, numpy=self.numpy)
    else:
        tp = False
    return (cont, tp)