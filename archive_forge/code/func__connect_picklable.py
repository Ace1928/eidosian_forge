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
def _connect_picklable(self, signal, func):
    """
        Like `.connect`, but the callback is kept when pickling/unpickling.

        Currently internal-use only.
        """
    cid = self.connect(signal, func)
    self._pickled_cids.add(cid)
    return cid