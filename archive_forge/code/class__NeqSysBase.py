from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
class _NeqSysBase(object):
    """ Baseclass for system of non-linear equations.

    This class contains shared logic used by its subclasses and is not meant to be used
    by end-users directly.
    """

    def __init__(self, names=None, param_names=None, x_by_name=None, par_by_name=None, latex_names=None, latex_param_names=None):
        self.names = names or ()
        self.param_names = param_names or ()
        self.x_by_name = x_by_name
        self.par_by_name = par_by_name
        self.latex_names = latex_names or ()
        self.latex_param_names = latex_param_names or ()

    def _get_solver_cb(self, solver, attached_solver):
        if attached_solver is not None:
            if solver is not None:
                raise ValueError('solver must be None.')
            solver = attached_solver(self)
        if callable(solver):
            return solver
        if solver is None:
            solver = os.environ.get('PYNEQSYS_SOLVER', 'scipy')
        return getattr(self, '_solve_' + solver)

    def rms(self, x, params=()):
        """ Returns root mean square value of f(x, params) """
        internal_x, internal_params = self.pre_process(np.asarray(x), np.asarray(params))
        if internal_params.ndim > 1:
            raise NotImplementedError('Parameters should be constant.')
        result = np.empty(internal_x.size // self.nx)
        for idx in range(internal_x.shape[0]):
            result[idx] = np.sqrt(np.mean(np.square(self.f_cb(internal_x[idx, :], internal_params))))
        return result

    def solve_series(self, x0, params, varied_data, varied_idx, internal_x0=None, solver=None, propagate=True, **kwargs):
        """ Solve system for a set of parameters in which one is varied

        Parameters
        ----------
        x0 : array_like
            Guess (subject to ``self.post_processors``)
        params : array_like
            Parameter values
        vaired_data : array_like
            Numerical values of the varied parameter.
        varied_idx : int or str
            Index of the varied parameter (indexing starts at 0).
            If ``self.par_by_name`` this should be the name (str) of the varied
            parameter.
        internal_x0 : array_like (default: None)
            Guess (*not* subject to ``self.post_processors``).
            Overrides ``x0`` when given.
        solver : str or callback
            See :meth:`solve`.
        propagate : bool (default: True)
            Use last successful solution as ``x0`` in consecutive solves.
        \\*\\*kwargs :
            Keyword arguments pass along to :meth:`solve`.

        Returns
        -------
        xout : array
            Of shape ``(varied_data.size, x0.size)``.
        info_dicts : list of dictionaries
             Dictionaries each containing keys such as containing 'success', 'nfev', 'njev' etc.

        """
        if self.x_by_name and isinstance(x0, dict):
            x0 = [x0[k] for k in self.names]
        if self.par_by_name:
            if isinstance(params, dict):
                params = [params[k] for k in self.param_names]
            if isinstance(varied_idx, str):
                varied_idx = self.param_names.index(varied_idx)
        new_params = np.atleast_1d(np.array(params, dtype=np.float64))
        xout = np.empty((len(varied_data), len(x0)))
        self.internal_xout = np.empty_like(xout)
        self.internal_params_out = np.empty((len(varied_data), len(new_params)))
        info_dicts = []
        new_x0 = np.array(x0, dtype=np.float64)
        conds = kwargs.get('initial_conditions', None)
        for idx, value in enumerate(varied_data):
            try:
                new_params[varied_idx] = value
            except TypeError:
                new_params = value
            if conds is not None:
                kwargs['initial_conditions'] = conds
            x, info_dict = self.solve(new_x0, new_params, internal_x0, solver, **kwargs)
            if propagate:
                if info_dict['success']:
                    try:
                        new_x0 = info_dict['x_vecs'][0]
                        internal_x0 = info_dict['internal_x_vecs'][0]
                        conds = info_dict['intermediate_info'][0].get('conditions', None)
                    except:
                        new_x0 = x
                        internal_x0 = None
                        conds = info_dict.get('conditions', None)
            xout[idx, :] = x
            self.internal_xout[idx, :] = self.internal_x
            self.internal_params_out[idx, :] = self.internal_params
            info_dicts.append(info_dict)
        return (xout, info_dicts)

    def plot_series(self, xres, varied_data, varied_idx, **kwargs):
        """ Plots the results from :meth:`solve_series`.

        Parameters
        ----------
        xres : array
            Of shape ``(varied_data.size, self.nx)``.
        varied_data : array
            See :meth:`solve_series`.
        varied_idx : int or str
            See :meth:`solve_series`.
        \\*\\*kwargs :
            Keyword arguments passed to :func:`pyneqsys.plotting.plot_series`.

        """
        for attr in 'names latex_names'.split():
            if kwargs.get(attr, None) is None:
                kwargs[attr] = getattr(self, attr)
        ax = plot_series(xres, varied_data, **kwargs)
        if self.par_by_name and isinstance(varied_idx, str):
            varied_idx = self.param_names.index(varied_idx)
        if self.latex_param_names:
            ax.set_xlabel('$%s$' % self.latex_param_names[varied_idx])
        elif self.param_names:
            ax.set_xlabel(self.param_names[varied_idx])
        return ax

    def plot_series_residuals(self, xres, varied_data, varied_idx, params, **kwargs):
        """ Analogous to :meth:`plot_series` but will plot residuals. """
        nf = len(self.f_cb(*self.pre_process(xres[0], params)))
        xerr = np.empty((xres.shape[0], nf))
        new_params = np.array(params)
        for idx, row in enumerate(xres):
            new_params[varied_idx] = varied_data[idx]
            xerr[idx, :] = self.f_cb(*self.pre_process(row, params))
        return self.plot_series(xerr, varied_data, varied_idx, **kwargs)

    def plot_series_residuals_internal(self, varied_data, varied_idx, **kwargs):
        """ Analogous to :meth:`plot_series` but for internal residuals from last run. """
        nf = len(self.f_cb(*self.pre_process(self.internal_xout[0], self.internal_params_out[0])))
        xerr = np.empty((self.internal_xout.shape[0], nf))
        for idx, (res, params) in enumerate(zip(self.internal_xout, self.internal_params_out)):
            xerr[idx, :] = self.f_cb(res, params)
        return self.plot_series(xerr, varied_data, varied_idx, **kwargs)

    def solve_and_plot_series(self, x0, params, varied_data, varied_idx, solver=None, plot_kwargs=None, plot_residuals_kwargs=None, **kwargs):
        """ Solve and plot for a series of a varied parameter.

        Convenience method, see :meth:`solve_series`, :meth:`plot_series` &
        :meth:`plot_series_residuals_internal` for more information.
        """
        sol, nfo = self.solve_series(x0, params, varied_data, varied_idx, solver=solver, **kwargs)
        ax_sol = self.plot_series(sol, varied_data, varied_idx, info=nfo, **plot_kwargs or {})
        extra = dict(ax_sol=ax_sol, info=nfo)
        if plot_residuals_kwargs:
            extra['ax_resid'] = self.plot_series_residuals_internal(varied_data, varied_idx, info=nfo, **plot_residuals_kwargs or {})
        return (sol, extra)