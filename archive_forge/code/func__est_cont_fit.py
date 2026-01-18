import numpy as np
from scipy import stats
from .distparams import distcont
def _est_cont_fit():
    for distname, arg in distcont:
        yield (check_cont_fit, distname, arg)