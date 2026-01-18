import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _compute_conf_interval(data, med, iqr, bootstrap):
    if bootstrap is not None:
        CI = _bootstrap_median(data, N=bootstrap)
        notch_min = CI[0]
        notch_max = CI[1]
    else:
        N = len(data)
        notch_min = med - 1.57 * iqr / np.sqrt(N)
        notch_max = med + 1.57 * iqr / np.sqrt(N)
    return (notch_min, notch_max)