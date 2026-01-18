from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
class PartiallySolvedSystem(SymbolicSys):
    """ Use analytic expressions for some dependent variables

    Parameters
    ----------
    original_system : SymbolicSys
    analytic_factory : callable
        User provided callback for expressing analytic solutions to a set of
        dependent variables in ``original_system``. The callback should have
        the signature: ``my_factory(x0, y0, p0, backend) -> dict``, where the returned
        dictionary maps dependent variabels (from ``original_system.dep``)
        to new expressions in remaining variables and initial conditions.
    \\*\\*kwargs : dict
        Keyword arguments passed onto :class:`SymbolicSys`.

    Attributes
    ----------
    free_names : list of str
    analytic_exprs : list of expressions
    analytic_cb : callback
    original_dep : dependent variable of original system

    Examples
    --------
    >>> odesys = SymbolicSys.from_callback(
    ...     lambda x, y, p: [
    ...         -p[0]*y[0],
    ...         p[0]*y[0] - p[1]*y[1]
    ...     ], 2, 2)
    >>> dep0 = odesys.dep[0]
    >>> partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be: {
    ...         dep0: y0[0]*be.exp(-p0[0]*(odesys.indep-x0))
    ...     })
    >>> print(partsys.exprs)  # doctest: +SKIP
    (_Dummy_29*p_0*exp(-p_0*(-_Dummy_28 + x)) - p_1*y_1,)
    >>> y0, k = [3, 2], [3.5, 2.5]
    >>> xout, yout, info = partsys.integrate([0, 1], y0, k, integrator='scipy')
    >>> info['success'], yout.shape[1]
    (True, 2)

    """
    _attrs_to_copy = SymbolicSys._attrs_to_copy + ('free_names', 'free_latex_names', 'original_dep')

    def __init__(self, original_system, analytic_factory, **kwargs):
        self._ori_sys = original_system
        self.analytic_factory = _ensure_4args(analytic_factory)
        roots = kwargs.pop('roots', self._ori_sys.roots)
        _be = self._ori_sys.be
        init_indep = self._ori_sys.init_indep or self._mk_init_indep(name=self._ori_sys.indep, be=self._ori_sys.be)
        init_dep = self._ori_sys.init_dep or self._ori_sys._mk_init_dep()
        if 'pre_processors' in kwargs or 'post_processors' in kwargs:
            raise NotImplementedError('Cannot override pre-/postprocessors')
        if 'backend' in kwargs and Backend(kwargs['backend']) != _be:
            raise ValueError('Cannot mix backends.')
        _pars = self._ori_sys.params
        if self._ori_sys.par_by_name:
            _pars = dict(zip(self._ori_sys.param_names, _pars))
        self.original_dep = self._ori_sys.dep
        _dep0 = dict(zip(self.original_dep, init_dep)) if self._ori_sys.dep_by_name else init_dep
        self.analytic_exprs = self.analytic_factory(init_indep, _dep0, _pars, _be)
        if len(self.analytic_exprs) == 0:
            raise ValueError('Failed to produce any analytic expressions.')
        new_dep = []
        free_names = []
        free_latex_names = []
        for idx, dep in enumerate(self.original_dep):
            if dep not in self.analytic_exprs:
                new_dep.append(dep)
                if self._ori_sys.names is not None and len(self._ori_sys.names) > 0:
                    free_names.append(self._ori_sys.names[idx])
                if self._ori_sys.latex_names is not None and len(self._ori_sys.latex_names) > 0:
                    free_latex_names.append(self._ori_sys.latex_names[idx])
        self.free_names = None if self._ori_sys.names is None else free_names
        self.free_latex_names = None if self._ori_sys.latex_names is None else free_latex_names
        self.append_iv = kwargs.get('append_iv', False)
        new_pars = _append(self._ori_sys.params, (init_indep,), init_dep)
        self.analytic_cb = self._get_analytic_callback(self._ori_sys, list(self.analytic_exprs.values()), new_dep, new_pars)
        self.ori_analyt_idx_map = OrderedDict([(self.original_dep.index(dep), idx) for idx, dep in enumerate(self.analytic_exprs)])
        self.ori_remaining_idx_map = {self.original_dep.index(dep): idx for idx, dep in enumerate(new_dep)}
        new_exprs = [expr.subs(self.analytic_exprs) for idx, expr in enumerate(self._ori_sys.exprs) if idx not in self.ori_analyt_idx_map]
        new_roots = None if roots is None else [expr.subs(self.analytic_exprs) for expr in roots]
        new_kw = kwargs.copy()
        for attr in self._attrs_to_copy:
            if attr not in new_kw and getattr(self._ori_sys, attr, None) is not None:
                new_kw[attr] = getattr(self._ori_sys, attr)
        if 'lower_bounds' not in new_kw and getattr(self._ori_sys, 'lower_bounds', None) is not None:
            new_kw['lower_bounds'] = _skip(self.ori_analyt_idx_map, self._ori_sys.lower_bounds)
        if 'upper_bounds' not in new_kw and getattr(self._ori_sys, 'upper_bounds', None) is not None:
            new_kw['upper_bounds'] = _skip(self.ori_analyt_idx_map, self._ori_sys.upper_bounds)
        if kwargs.get('linear_invariants', None) is None:
            if new_kw.get('linear_invariants', None) is not None:
                if new_kw['linear_invariants'].shape[1] != self._ori_sys.ny:
                    raise ValueError('Unexpected number of columns in original linear_invariants.')
                new_kw['linear_invariants'] = new_kw['linear_invariants'][:, [i for i in range(self._ori_sys.ny) if i not in self.ori_analyt_idx_map]]

        def partially_solved_pre_processor(x, y, p):
            if y.ndim == 2:
                return zip(*[partially_solved_pre_processor(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
            return (x, _skip(self.ori_analyt_idx_map, y), _append(p, [x[0]], y))

        def partially_solved_post_processor(x, y, p):
            try:
                y[0][0, 0]
            except:
                pass
            else:
                return zip(*[partially_solved_post_processor(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
            new_y = np.empty(y.shape[:-1] + (y.shape[-1] + len(self.analytic_exprs),))
            analyt_y = self.analytic_cb(x, y, p)
            for idx in range(self._ori_sys.ny):
                if idx in self.ori_analyt_idx_map:
                    new_y[..., idx] = analyt_y[..., self.ori_analyt_idx_map[idx]]
                else:
                    new_y[..., idx] = y[..., self.ori_remaining_idx_map[idx]]
            return (x, new_y, p[:-(1 + self._ori_sys.ny)])
        new_kw['pre_processors'] = self._ori_sys.pre_processors + [partially_solved_pre_processor]
        new_kw['post_processors'] = [partially_solved_post_processor] + self._ori_sys.post_processors
        super(PartiallySolvedSystem, self).__init__(zip(new_dep, new_exprs), self._ori_sys.indep, new_pars, backend=_be, roots=new_roots, init_indep=init_indep, init_dep=init_dep, **new_kw)

    @classmethod
    def from_linear_invariants(cls, ori_sys, preferred=None, **kwargs):
        """ Reformulates the ODE system in fewer variables.

        Given linear invariant equations one can always reduce the number
        of dependent variables in the system by the rank of the matrix describing
        this linear system.

        Parameters
        ----------
        ori_sys : :class:`SymbolicSys` instance
        preferred : iterable of preferred dependent variables
            Due to numerical rounding it is preferable to choose the variables
            which are expected to be of the largest magnitude during integration.
        \\*\\*kwargs :
            Keyword arguments passed on to constructor.
        """
        _be = ori_sys.be
        A = _be.Matrix(ori_sys.linear_invariants)
        rA, pivots = A.rref()
        if len(pivots) < A.shape[0]:
            raise NotImplementedError('Linear invariants contain linear dependencies.')
        per_row_cols = [(ri, [ci for ci in range(A.cols) if A[ri, ci] != 0]) for ri in range(A.rows)]
        if preferred is None:
            preferred = ori_sys.names[:A.rows] if ori_sys.dep_by_name else list(range(A.rows))
        targets = [ori_sys.names.index(dep) if ori_sys.dep_by_name else dep if isinstance(dep, int) else ori_sys.dep.index(dep) for dep in preferred]
        row_tgt = []
        for ri, colids in sorted(per_row_cols, key=lambda k: len(k[1])):
            for tgt in targets:
                if tgt in colids:
                    row_tgt.append((ri, tgt))
                    targets.remove(tgt)
                    break
            if len(targets) == 0:
                break
        else:
            raise ValueError('Could not find a solutions for: %s' % targets)

        def analytic_factory(x0, y0, p0, be):
            return {ori_sys.dep[tgt]: y0[ori_sys.dep[tgt] if ori_sys.dep_by_name else tgt] - sum([A[ri, ci] * (ori_sys.dep[ci] - y0[ori_sys.dep[ci] if ori_sys.dep_by_name else ci]) for ci in range(A.cols) if ci != tgt]) / A[ri, tgt] for ri, tgt in row_tgt}
        ori_li_nms = ori_sys.linear_invariant_names or ()
        new_lin_invar = [[cell for ci, cell in enumerate(row) if ci not in list(zip(*row_tgt))[1]] for ri, row in enumerate(A.tolist()) if ri not in list(zip(*row_tgt))[0]]
        new_lin_i_nms = [nam for ri, nam in enumerate(ori_li_nms) if ri not in list(zip(*row_tgt))[0]]
        return cls(ori_sys, analytic_factory, linear_invariants=new_lin_invar, linear_invariant_names=new_lin_i_nms, **kwargs)

    @staticmethod
    def _get_analytic_callback(ori_sys, analytic_exprs, new_dep, new_params):
        return _Callback(ori_sys.indep, new_dep, new_params, analytic_exprs, Lambdify=ori_sys.be.Lambdify)

    def __getitem__(self, key):
        ori_dep = self.original_dep[self.names.index(key)]
        return self.analytic_exprs.get(ori_dep, ori_dep)

    def integrate(self, *args, **kwargs):
        if 'atol' in kwargs:
            atol = kwargs.pop('atol')
            if isinstance(atol, dict):
                atol = [atol[k] for k in self.free_names]
            else:
                try:
                    len(atol)
                except TypeError:
                    pass
                else:
                    atol = [atol[idx] for idx in _skip(self.ori_analyt_idx_map, atol)]
            kwargs['atol'] = atol
        return super(PartiallySolvedSystem, self).integrate(*args, **kwargs)