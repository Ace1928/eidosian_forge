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
def addInPlace(self, other):
    if isinstance(other, LpConstraint):
        if self.sense * other.sense >= 0:
            LpAffineExpression.addInPlace(self, other)
            self.sense |= other.sense
        else:
            LpAffineExpression.subInPlace(self, other)
            self.sense |= -other.sense
    elif isinstance(other, list):
        for e in other:
            self.addInPlace(e)
    else:
        LpAffineExpression.addInPlace(self, other)
    return self