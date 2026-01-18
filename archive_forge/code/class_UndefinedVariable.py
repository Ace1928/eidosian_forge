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
class UndefinedVariable:

    def __init__(self):
        raise NotImplementedError('Not intended for instantiation')
    target = ir.UNDEFINED