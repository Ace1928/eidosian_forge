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
def addVariables(self, variables):
    """
        Adds variables to the problem before a constraint is added

        @param variables: the variables to be added
        """
    for v in variables:
        self.addVariable(v)