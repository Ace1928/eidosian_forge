import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class ForceLiteralArg(NumbaError):
    """A Pseudo-exception to signal the dispatcher to type an argument literally

    Attributes
    ----------
    requested_args : frozenset[int]
        requested positions of the arguments.
    """

    def __init__(self, arg_indices, fold_arguments=None, loc=None):
        """
        Parameters
        ----------
        arg_indices : Sequence[int]
            requested positions of the arguments.
        fold_arguments: callable
            A function ``(tuple, dict) -> tuple`` that binds and flattens
            the ``args`` and ``kwargs``.
        loc : numba.ir.Loc or None
        """
        super(ForceLiteralArg, self).__init__('Pseudo-exception to force literal arguments in the dispatcher', loc=loc)
        self.requested_args = frozenset(arg_indices)
        self.fold_arguments = fold_arguments

    def bind_fold_arguments(self, fold_arguments):
        """Bind the fold_arguments function
        """
        from numba.core.utils import chain_exception
        e = ForceLiteralArg(self.requested_args, fold_arguments, loc=self.loc)
        return chain_exception(e, self)

    def combine(self, other):
        """Returns a new instance by or'ing the requested_args.
        """
        if not isinstance(other, ForceLiteralArg):
            m = '*other* must be a {} but got a {} instead'
            raise TypeError(m.format(ForceLiteralArg, type(other)))
        return ForceLiteralArg(self.requested_args | other.requested_args)

    def __or__(self, other):
        """Same as self.combine(other)
        """
        return self.combine(other)