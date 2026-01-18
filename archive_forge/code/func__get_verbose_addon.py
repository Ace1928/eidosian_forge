import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _get_verbose_addon(qc_dict):
    alpha = qc_dict['alpha']
    params = qc_dict['params']
    fprime = qc_dict['fprime']
    passed_array = qc_dict['passed_array']
    addon = '\n------ verbose QC printout -----------------'
    addon = '\n------ Recall the problem was rescaled by 1 / nobs ---'
    addon += '\n|%-10s|%-10s|%-10s|%-10s|' % ('passed', 'alpha', 'fprime', 'param')
    addon += '\n--------------------------------------------'
    for i in range(len(alpha)):
        addon += '\n|%-10s|%-10.3e|%-10.3e|%-10.3e|' % (passed_array[i], alpha[i], fprime[i], params[i])
    return addon