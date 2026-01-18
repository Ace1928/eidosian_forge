from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
class ScaledSys(TransformedSys):
    """ Transformed system where the variables have been scaled linearly.

    Parameters
    ----------
    dep_exprs : iterable of (symbol, expression)-pairs
        see :class:`SymbolicSys`
    indep : Symbol
        see :class:`SymbolicSys`
    dep_scaling : number (>0) or iterable of numbers
        scaling of the dependent variables (default: 1)
    indep_scaling : number (>0)
        scaling of the independent variable (default: 1)
    params :
        see :class:`SymbolicSys`
    \\*\\*kwargs :
        Keyword arguments passed onto :class:`TransformedSys`.

    Examples
    --------
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> scaled1 = ScaledSys([(x, x*x)], dep_scaling=1000)
    >>> scaled1.exprs == (x**2/1000,)
    True
    >>> scaled2 = ScaledSys([(x, x**3)], dep_scaling=1000)
    >>> scaled2.exprs == (x**3/1000000,)
    True

    """

    @staticmethod
    def _scale_fw_bw(scaling):
        return (lambda x: scaling * x, lambda x: x / scaling)

    def __init__(self, dep_exprs, indep=None, dep_scaling=1, indep_scaling=1, params=(), **kwargs):
        dep_exprs = list(dep_exprs)
        dep, exprs = list(zip(*dep_exprs))
        try:
            n = len(dep_scaling)
        except TypeError:
            n = len(dep_exprs)
            dep_scaling = [dep_scaling] * n
        transf_dep_cbs = [self._scale_fw_bw(s) for s in dep_scaling]
        transf_indep_cbs = self._scale_fw_bw(indep_scaling)
        super(ScaledSys, self).__init__(dep_exprs, indep, params=params, dep_transf=[(transf_cb[0](depi), transf_cb[1](depi)) for transf_cb, depi in zip(transf_dep_cbs, dep)], indep_transf=(transf_indep_cbs[0](indep), transf_indep_cbs[1](indep)) if indep is not None else None, **kwargs)

    @classmethod
    def from_callback(cls, cb, ny=None, nparams=None, dep_scaling=1, indep_scaling=1, **kwargs):
        """
        Create an instance from a callback.

        Analogous to :func:`SymbolicSys.from_callback`.

        Parameters
        ----------
        cb : callable
            Signature rhs(x, y[:], p[:]) -> f[:]
        ny : int
            length of y
        nparams : int
            length of p
        dep_scaling : number (>0) or iterable of numbers
            scaling of the dependent variables (default: 1)
        indep_scaling: number (>0)
            scaling of the independent variable (default: 1)
        \\*\\*kwargs :
            Keyword arguments passed onto :class:`ScaledSys`.

        Examples
        --------
        >>> def f(x, y, p):
        ...     return [p[0]*y[0]**2]
        >>> odesys = ScaledSys.from_callback(f, 1, 1, dep_scaling=10)
        >>> odesys.exprs
        (p_0*y_0**2/10,)

        """
        return TransformedSys.from_callback(cb, ny, nparams, dep_transf_cbs=repeat(cls._scale_fw_bw(dep_scaling)), indep_transf_cbs=cls._scale_fw_bw(indep_scaling), **kwargs)