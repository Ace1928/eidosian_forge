import numpy as np
import warnings
from scipy.optimize import minimize
from ase.parallel import world
from ase.io.jsonio import write_json
from ase.optimize.optimize import Optimizer
from ase.optimize.gpmin.gp import GaussianProcess
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior
def acquisition(self, r):
    e = self.predict(r)
    return (e[0], e[1:])