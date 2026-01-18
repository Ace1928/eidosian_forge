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
def _stmt_index(self, defstmt, block, stop=-1):
    """Find the positional index of the statement at ``block``.

        Assumptions:
        - no two statements can point to the same object.
        """
    for i in range(len(block.body))[:stop]:
        if block.body[i] is defstmt:
            return i
    return len(block.body)