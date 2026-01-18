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
class OverloadSelector(object):
    """
    An object matching an actual signature against a registry of formal
    signatures and choosing the best candidate, if any.

    In the current implementation:
    - a "signature" is a tuple of type classes or type instances
    - the "best candidate" is the most specific match
    """

    def __init__(self):
        self.versions = []
        self._cache = {}

    def find(self, sig):
        out = self._cache.get(sig)
        if out is None:
            out = self._find(sig)
            self._cache[sig] = out
        return out

    def _find(self, sig):
        candidates = self._select_compatible(sig)
        if candidates:
            return candidates[self._best_signature(candidates)]
        else:
            raise errors.NumbaNotImplementedError(f'{self}, {sig}')

    def _select_compatible(self, sig):
        """
        Select all compatible signatures and their implementation.
        """
        out = {}
        for ver_sig, impl in self.versions:
            if self._match_arglist(ver_sig, sig):
                out[ver_sig] = impl
        return out

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

    def _sort_signatures(self, candidates):
        """
        Sort signatures in ascending level of genericity.

        Returns a 2-tuple:

            * ordered list of signatures
            * dictionary containing genericity scores
        """
        genericity = defaultdict(int)
        for this, other in permutations(candidates.keys(), r=2):
            matched = self._match_arglist(formal_args=this, actual_args=other)
            if matched:
                genericity[this] += 1
        ordered = sorted(candidates.keys(), key=lambda x: genericity[x])
        return (ordered, genericity)

    def _match_arglist(self, formal_args, actual_args):
        """
        Returns True if the signature is "matching".
        A formal signature is "matching" if the actual signature matches exactly
        or if the formal signature is a compatible generic signature.
        """
        if formal_args and isinstance(formal_args[-1], types.VarArg):
            ndiff = len(actual_args) - len(formal_args) + 1
            formal_args = formal_args[:-1] + (formal_args[-1].dtype,) * ndiff
        if len(formal_args) != len(actual_args):
            return False
        for formal, actual in zip(formal_args, actual_args):
            if not self._match(formal, actual):
                return False
        return True

    def _match(self, formal, actual):
        if formal == actual:
            return True
        elif types.Any == formal:
            return True
        elif isinstance(formal, type) and issubclass(formal, types.Type):
            if isinstance(actual, type) and issubclass(actual, formal):
                return True
            elif isinstance(actual, formal):
                return True

    def append(self, value, sig):
        """
        Add a formal signature and its associated value.
        """
        assert isinstance(sig, tuple), (value, sig)
        self.versions.append((sig, value))
        self._cache.clear()