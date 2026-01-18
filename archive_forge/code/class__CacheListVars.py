import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
class _CacheListVars:

    def __init__(self):
        self._saved = {}

    def get(self, inst):
        got = self._saved.get(inst)
        if got is None:
            self._saved[inst] = got = inst.list_vars()
        return got