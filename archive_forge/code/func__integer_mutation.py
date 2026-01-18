import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
def _integer_mutation(self):
    n_I_R = self.i_I_R.shape[0]
    if n_I_R == 0:
        return numpy.zeros((self.lambda_, self.dim))
    elif n_I_R == self.dim:
        p = self.lambda_ / 2.0 / self.lambda_
    else:
        p = min(self.lambda_ / 2.0, self.lambda_ / 10.0 + n_I_R / self.dim) / self.lambda_
    Rp = numpy.zeros((self.lambda_, self.dim))
    Rpp = numpy.zeros((self.lambda_, self.dim))
    for i, j in zip(range(self.lambda_), cycle(self.i_I_R)):
        if numpy.random.rand() < p:
            Rp[i, j] = 1
            Rpp[i, j] = numpy.random.geometric(p=0.7 ** (1.0 / n_I_R)) - 1
    I_pm1 = (-1) ** numpy.random.randint(0, 2, (self.lambda_, self.dim))
    R_int = I_pm1 * (Rp + Rpp)
    return R_int