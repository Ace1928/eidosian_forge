import warnings
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
Old BFGS behaviour for scaling step lengths

        This keeps the behaviour of truncating individual steps. Some might
        depend of this as some absurd kind of stimulated annealing to find the
        global minimum.
        