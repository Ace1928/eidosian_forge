from __future__ import absolute_import, division, print_function
from itertools import chain
import numpy as np
from sym import Backend
from sym.util import banded_jacobian, check_transforms
from .core import NeqSys, _ensure_3args
class TransformedSys(SymbolicSys):
    """ A system which transforms the equations and variables internally

    Can be used to reformulate a problem in a numerically more stable form.

    Parameters
    ----------
    x : iterable of variables
    exprs : iterable of expressions
         Expressions to find root for (untransformed).
    transf : iterable of pairs of expressions
        Forward, backward transformed instances of x.
    params : iterable of symbols
    post_adj : callable (default: None)
        To tweak expression after transformation.
    \\*\\*kwargs :
        Keyword arguments passed onto :class:`SymbolicSys`.

    """
    _use_symbol_latex_names = False

    def __init__(self, x, exprs, transf, params=(), post_adj=None, **kwargs):
        self.fw, self.bw = zip(*transf)
        check_transforms(self.fw, self.bw, x)
        exprs = [e.subs(zip(x, self.fw)) for e in exprs]
        super(TransformedSys, self).__init__(x, _map2l(post_adj, exprs), params, pre_processors=[lambda xarr, params: (self.bw_cb(xarr), params)], post_processors=[lambda xarr, params: (self.fw_cb(xarr), params)], **kwargs)
        self.fw_cb = self.be.Lambdify(x, self.fw)
        self.bw_cb = self.be.Lambdify(x, self.bw)

    @classmethod
    def from_callback(cls, cb, transf_cbs, nx, nparams=0, pre_adj=None, **kwargs):
        """ Generate a TransformedSys instance from a callback

        Parameters
        ----------
        cb : callable
            Should have the signature ``cb(x, p, backend) -> list of exprs``.
            The callback ``cb`` should return *untransformed* expressions.
        transf_cbs : pair or iterable of pairs of callables
            Callables for forward- and backward-transformations. Each
            callable should take a single parameter (expression) and
            return a single expression.
        nx : int
            Number of unkowns.
        nparams : int
            Number of parameters.
        pre_adj : callable, optional
            To tweak expression prior to transformation. Takes a
            sinlge argument (expression) and return a single argument
            rewritten expression.
        \\*\\*kwargs :
            Keyword arguments passed on to :class:`TransformedSys`. See also
            :class:`SymbolicSys` and :class:`pyneqsys.NeqSys`.

        Examples
        --------
        >>> import sympy as sp
        >>> transformed = TransformedSys.from_callback(lambda x, p, be: [
        ...     x[0]*x[1] - p[0],
        ...     be.exp(-x[0]) + be.exp(-x[1]) - p[0]**-2
        ... ], (sp.log, sp.exp), 2, 1)
        ...


        """
        be = Backend(kwargs.pop('backend', None))
        x = be.real_symarray('x', nx)
        p = be.real_symarray('p', nparams)
        try:
            transf = [(transf_cbs[idx][0](xi), transf_cbs[idx][1](xi)) for idx, xi in enumerate(x)]
        except TypeError:
            transf = zip(_map2(transf_cbs[0], x), _map2(transf_cbs[1], x))
        try:
            exprs = cb(x, p, be)
        except TypeError:
            exprs = _ensure_3args(cb)(x, p, be)
        return cls(x, _map2l(pre_adj, exprs), transf, p, backend=be, **kwargs)