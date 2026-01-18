import inspect
import numpy as np
from ._optimize import OptimizeWarning, OptimizeResult
from warnings import warn
from ._highs._highs_wrapper import _highs_wrapper
from ._highs._highs_constants import (
from scipy.sparse import csc_matrix, vstack, issparse
def _highs_to_scipy_status_message(highs_status, highs_message):
    """Converts HiGHS status number/message to SciPy status number/message"""
    scipy_statuses_messages = {None: (4, 'HiGHS did not provide a status code. '), MODEL_STATUS_NOTSET: (4, ''), MODEL_STATUS_LOAD_ERROR: (4, ''), MODEL_STATUS_MODEL_ERROR: (2, ''), MODEL_STATUS_PRESOLVE_ERROR: (4, ''), MODEL_STATUS_SOLVE_ERROR: (4, ''), MODEL_STATUS_POSTSOLVE_ERROR: (4, ''), MODEL_STATUS_MODEL_EMPTY: (4, ''), MODEL_STATUS_RDOVUB: (4, ''), MODEL_STATUS_REACHED_OBJECTIVE_TARGET: (4, ''), MODEL_STATUS_OPTIMAL: (0, 'Optimization terminated successfully. '), MODEL_STATUS_REACHED_TIME_LIMIT: (1, 'Time limit reached. '), MODEL_STATUS_REACHED_ITERATION_LIMIT: (1, 'Iteration limit reached. '), MODEL_STATUS_INFEASIBLE: (2, 'The problem is infeasible. '), MODEL_STATUS_UNBOUNDED: (3, 'The problem is unbounded. '), MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE: (4, 'The problem is unbounded or infeasible. ')}
    unrecognized = (4, 'The HiGHS status code was not recognized. ')
    scipy_status, scipy_message = scipy_statuses_messages.get(highs_status, unrecognized)
    scipy_message = f'{scipy_message}(HiGHS Status {highs_status}: {highs_message})'
    return (scipy_status, scipy_message)