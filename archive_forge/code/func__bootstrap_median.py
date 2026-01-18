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
def _bootstrap_median(data, N=5000):
    M = len(data)
    percentiles = [2.5, 97.5]
    bs_index = np.random.randint(M, size=(N, M))
    bsData = data[bs_index]
    estimate = np.median(bsData, axis=1, overwrite_input=True)
    CI = np.percentile(estimate, percentiles)
    return CI