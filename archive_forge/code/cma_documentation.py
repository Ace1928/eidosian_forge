import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
Update the current covariance matrix strategy from the *population*.
        :param population: A list of individuals from which to update the
                           parameters.
        