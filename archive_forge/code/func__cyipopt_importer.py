import abc
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
def _cyipopt_importer():
    import cyipopt
    if not hasattr(cyipopt, 'Problem'):
        cyipopt.Problem = cyipopt.problem
    if not hasattr(cyipopt, '__version__'):
        import ipopt
        cyipopt.__version__ = ipopt.__version__
    if not hasattr(cyipopt, 'STATUS_MESSAGES'):
        import ipopt_wrapper
        cyipopt.STATUS_MESSAGES = ipopt_wrapper.STATUS_MESSAGES
    return cyipopt