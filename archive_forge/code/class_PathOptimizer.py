import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
class PathOptimizer(object):
    '''Base class for different path optimizers to inherit from.

    Subclassed optimizers should define a call method with signature::

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            """
            Parameters
            ----------
            inputs : list[set[str]]
                The indices of each input array.
            outputs : set[str]
                The output indices
            size_dict : dict[str, int]
                The size of each index
            memory_limit : int, optional
                If given, the maximum allowed memory.
            """
            # ... compute path here ...
            return path

    where ``path`` is a list of int-tuples specifiying a contraction order.
    '''

    def _check_args_against_first_call(self, inputs, output, size_dict):
        """Utility that stateful optimizers can use to ensure they are not
        called with different contractions across separate runs.
        """
        args = (inputs, output, size_dict)
        if not hasattr(self, '_first_call_args'):
            self._first_call_args = args
        elif args != self._first_call_args:
            raise ValueError('The arguments specifiying the contraction that this path optimizer instance was called with have changed - try creating a new instance.')

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        raise NotImplementedError