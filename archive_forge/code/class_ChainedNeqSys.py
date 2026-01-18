from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
class ChainedNeqSys(_NeqSysBase):
    """ Chain multiple formulations of non-linear systems for using
    the result of one as starting guess for the other

    Examples
    --------
    >>> neqsys_lin = NeqSys(1, 1, lambda x, p: [x[0]**2 - p[0]])
    >>> from math import log, exp
    >>> neqsys_log = NeqSys(1, 1, lambda x, p: [2*x[0] - log(p[0])],
    ...    pre_processors=[lambda x, p: ([log(x[0]+1e-60)], p)],
    ...    post_processors=[lambda x, p: ([exp(x[0])], p)])
    >>> chained = ChainedNeqSys([neqsys_log, neqsys_lin])
    >>> x, info = chained.solve([1, 1], [4])
    >>> assert info['success']
    >>> print(x)  # doctest: +NORMALIZE_WHITESPACE
    [ 2.]
    >>> print(info['intermediate_info'][0]['nfev'],
    ...       info['intermediate_info'][1]['nfev'])  # doctest: +SKIP
    4 3

    """

    def __init__(self, neqsystems, **kwargs):
        super(ChainedNeqSys, self).__init__(**kwargs)
        self.neqsystems = neqsystems
        self.f_cb = self.neqsystems[0].f_cb

    def solve(self, x0, params=(), internal_x0=None, solver=None, **kwargs):
        x_vecs = []
        info_vec = []
        internal_x_vecs = []
        for idx, neqsys in enumerate(self.neqsystems):
            x0, info = neqsys.solve(x0, params, internal_x0, solver, **kwargs)
            if idx == 0:
                self.internal_x = info['x']
                self.internal_params = neqsys.internal_params
            internal_x0 = None
            if 'conditions' in info:
                kwargs['initial_conditions'] = info['conditions']
            x_vecs.append(x0)
            internal_x_vecs.append(neqsys.internal_x)
            info_vec.append(info)
        info = {'x': self.internal_x, 'success': info['success'], 'nfev': sum([nfo['nfev'] for nfo in info_vec]), 'njev': sum([nfo.get('njev', 0) for nfo in info_vec])}
        if 'fun' in info:
            info['fun'] = info['fun']
        info['x_vecs'] = x_vecs
        info['intermediate_info'] = info_vec
        info['internal_x_vecs'] = internal_x_vecs
        return (x0, info)
    solve.__doc__ = NeqSys.solve.__doc__

    def post_process(self, x, params):
        return self.neqsystems[0].post_process(x, params)
    post_process.__doc__ = NeqSys.post_process.__doc__

    def pre_process(self, x, params, conds=None):
        return self.neqsystems[0].pre_process(x, params)
    pre_process.__doc__ = NeqSys.pre_process.__doc__