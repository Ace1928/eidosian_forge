from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
def _best_signature(self, candidates):
    """
        Returns the best signature out of the candidates
        """
    ordered, genericity = self._sort_signatures(candidates)
    if len(ordered) > 1:
        firstscore = genericity[ordered[0]]
        same = list(takewhile(lambda x: genericity[x] == firstscore, ordered))
        if len(same) > 1:
            msg = ['{n} ambiguous signatures'.format(n=len(same))]
            for sig in same:
                msg += ['{0} => {1}'.format(sig, candidates[sig])]
            raise errors.NumbaTypeError('\n'.join(msg))
    return ordered[0]