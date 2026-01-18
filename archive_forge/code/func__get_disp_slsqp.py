import numpy as np
from scipy.optimize import fmin_slsqp
import statsmodels.base.l1_solvers_common as l1_solvers_common
def _get_disp_slsqp(disp, retall):
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0
    return disp_slsqp