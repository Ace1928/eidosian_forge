from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Iterable
import numbers
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
import cvxpy.interface as intf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import expression
from cvxpy.settings import (
def _get_attr_str(self) -> str:
    """Get a string representing the attributes.
        """
    attr_str = ''
    for attr, val in self.attributes.items():
        if attr != 'real' and val:
            attr_str += ', %s=%s' % (attr, val)
    return attr_str