from __future__ import absolute_import, division, print_function
from itertools import chain
import numpy as np
from sym import Backend
from sym.util import banded_jacobian, check_transforms
from .core import NeqSys, _ensure_3args
class SymbolicSys(NeqSys):
    """ Symbolically defined system of non-linear equations.

    This object is analogous to :class:`pyneqsys.NeqSys` but instead of
    providing a callable, the user provides symbolic expressions.

    Parameters
    ----------
    x : iterable of Symbols
    exprs : iterable of expressions for ``f``
    params : iterable of Symbols (optional)
        list of symbols appearing in exprs which are parameters
    jac : ImmutableMatrix or bool
        If ``True``:
            - Calculate Jacobian from ``exprs``.
        If ``False``:
            - Do not compute Jacobian (numeric approximation).
        If ImmutableMatrix:
            - User provided expressions for the Jacobian.
    backend : str or sym.Backend
        See documentation of `sym.Backend <https://pythonhosted.org/sym/sym.html#sym.backend.Backend>`_.
    module : str
        ``module`` keyword argument passed to ``backend.Lambdify``.
    \\*\\*kwargs:
        See :py:class:`pyneqsys.core.NeqSys`.

    Examples
    --------
    >>> import sympy as sp
    >>> e = sp.exp
    >>> x = x0, x1 = sp.symbols('x:2')
    >>> params = a, b = sp.symbols('a b')
    >>> neqsys = SymbolicSys(x, [a*(1 - x0), b*(x1 - x0**2)], params)
    >>> xout, sol = neqsys.solve('scipy', [-10, -5], [1, 10])
    >>> print(xout)  # doctest: +NORMALIZE_WHITESPACE
    [ 1.  1.]
    >>> print(neqsys.get_jac()[0, 0])
    -a

    Notes
    -----
    When using SymPy as the backend, a limited number of unknowns is supported.
    The reason is that (currently) ``sympy.lambdify`` has an upper limit on
    number of arguments.

    """

    def __init__(self, x, exprs, params=(), jac=True, backend=None, **kwargs):
        self.x = x
        self.exprs = exprs
        self.params = params
        self._jac = jac
        self.be = Backend(backend)
        self.nf, self.nx = (len(exprs), len(x))
        self.band = kwargs.get('band', None)
        self.module = kwargs.pop('module', 'numpy')
        super(SymbolicSys, self).__init__(self.nf, self.nx, self._get_f_cb(), self._get_j_cb(), **kwargs)

    @classmethod
    def from_callback(cls, cb, nx=None, nparams=None, **kwargs):
        """ Generate a SymbolicSys instance from a callback.

        Parameters
        ----------
        cb : callable
            Should have the signature ``cb(x, p, backend) -> list of exprs``.
        nx : int
            Number of unknowns, when not given it is deduced from ``kwargs['names']``.
        nparams : int
            Number of parameters, when not given it is deduced from ``kwargs['param_names']``.

        \\*\\*kwargs :
            Keyword arguments passed on to :class:`SymbolicSys`. See also :class:`pyneqsys.NeqSys`.

        Examples
        --------
        >>> symbolicsys = SymbolicSys.from_callback(lambda x, p, be: [
        ...     x[0]*x[1] - p[0],
        ...     be.exp(-x[0]) + be.exp(-x[1]) - p[0]**-2
        ... ], 2, 1)
        ...

        """
        if kwargs.get('x_by_name', False):
            if 'names' not in kwargs:
                raise ValueError('Need ``names`` in kwargs.')
            if nx is None:
                nx = len(kwargs['names'])
            elif nx != len(kwargs['names']):
                raise ValueError('Inconsistency between nx and length of ``names``.')
        if kwargs.get('par_by_name', False):
            if 'param_names' not in kwargs:
                raise ValueError('Need ``param_names`` in kwargs.')
            if nparams is None:
                nparams = len(kwargs['param_names'])
            elif nparams != len(kwargs['param_names']):
                raise ValueError('Inconsistency between ``nparam`` and length of ``param_names``.')
        if nparams is None:
            nparams = 0
        if nx is None:
            raise ValueError('Need ``nx`` of ``names`` together with ``x_by_name==True``.')
        be = Backend(kwargs.pop('backend', None))
        x = be.real_symarray('x', nx)
        p = be.real_symarray('p', nparams)
        _x = dict(zip(kwargs['names'], x)) if kwargs.get('x_by_name', False) else x
        _p = dict(zip(kwargs['param_names'], p)) if kwargs.get('par_by_name', False) else p
        try:
            exprs = cb(_x, _p, be)
        except TypeError:
            exprs = _ensure_3args(cb)(_x, _p, be)
        return cls(x, exprs, p, backend=be, **kwargs)

    def get_jac(self):
        """ Return the jacobian of the expressions """
        if self._jac is True:
            if self.band is None:
                f = self.be.Matrix(self.nf, 1, self.exprs)
                _x = self.be.Matrix(self.nx, 1, self.x)
                return f.jacobian(_x)
            else:
                return self.be.Matrix(banded_jacobian(self.exprs, self.x, *self.band))
        elif self._jac is False:
            return False
        else:
            return self._jac

    def _get_f_cb(self):
        args = list(chain(self.x, self.params))
        kw = dict(module=self.module, dtype=object if self.module == 'mpmath' else None)
        try:
            cb = self.be.Lambdify(args, self.exprs, **kw)
        except TypeError:
            cb = self.be.Lambdify(args, self.exprs)

        def f(x, params):
            return cb(np.concatenate((x, params), axis=-1))
        return f

    def _get_j_cb(self):
        args = list(chain(self.x, self.params))
        kw = dict(module=self.module, dtype=object if self.module == 'mpmath' else None)
        try:
            cb = self.be.Lambdify(args, self.get_jac(), **kw)
        except TypeError:
            cb = self.be.Lambdify(args, self.get_jac())

        def j(x, params):
            return cb(np.concatenate((x, params), axis=-1))
        return j
    _use_symbol_latex_names = True

    def _repr_latex_(self):
        from ._sympy import NeqSysTexPrinter
        if self.latex_names and (self.latex_param_names if len(self.params) else True):
            pretty = {s: n for s, n in chain(zip(self.x, self.latex_names) if self._use_symbol_latex_names else [], zip(self.params, self.latex_param_names))}
        else:
            pretty = {}
        return '$%s$' % NeqSysTexPrinter(dict(symbol_names=pretty)).doprint(self.exprs)