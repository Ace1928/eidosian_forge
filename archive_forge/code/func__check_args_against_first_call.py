import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def _check_args_against_first_call(self, inputs, output, size_dict):
    """Utility that stateful optimizers can use to ensure they are not
        called with different contractions across separate runs.
        """
    args = (inputs, output, size_dict)
    if not hasattr(self, '_first_call_args'):
        self._first_call_args = args
    elif args != self._first_call_args:
        raise ValueError('The arguments specifiying the contraction that this path optimizer instance was called with have changed - try creating a new instance.')