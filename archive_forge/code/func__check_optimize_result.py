import warnings
import numpy as np
import scipy
from ..exceptions import ConvergenceWarning
from .fixes import line_search_wolfe1, line_search_wolfe2
def _check_optimize_result(solver, result, max_iter=None, extra_warning_msg=None):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    solver : str
       Solver name. Currently only `lbfgs` is supported.

    result : OptimizeResult
       Result of the scipy.optimize.minimize function.

    max_iter : int, default=None
       Expected maximum number of iterations.

    extra_warning_msg : str, default=None
        Extra warning message.

    Returns
    -------
    n_iter : int
       Number of iterations.
    """
    if solver == 'lbfgs':
        if result.status != 0:
            try:
                result_message = result.message.decode('latin1')
            except AttributeError:
                result_message = result.message
            warning_msg = '{} failed to converge (status={}):\n{}.\n\nIncrease the number of iterations (max_iter) or scale the data as shown in:\n    https://scikit-learn.org/stable/modules/preprocessing.html'.format(solver, result.status, result_message)
            if extra_warning_msg is not None:
                warning_msg += '\n' + extra_warning_msg
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
        if max_iter is not None:
            n_iter_i = min(result.nit, max_iter)
        else:
            n_iter_i = result.nit
    else:
        raise NotImplementedError
    return n_iter_i