from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def addVariableToConstraints(self, e):
    """adds a variable to the constraints indicated by
        the LpConstraintVars in e
        """
    for constraint, coeff in e.items():
        constraint.addVariable(self, coeff)