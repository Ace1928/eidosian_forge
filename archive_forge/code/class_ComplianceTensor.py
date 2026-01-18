from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
class ComplianceTensor(Tensor):
    """
    This class represents the compliance tensor, and exists
    primarily to keep the voigt-conversion scheme consistent
    since the compliance tensor has a unique vscale.
    """

    def __new__(cls, s_array) -> Self:
        """
        Args:
            s_array ():
        """
        vscale = np.ones((6, 6))
        vscale[3:] *= 2
        vscale[:, 3:] *= 2
        obj = super().__new__(cls, s_array, vscale=vscale)
        return obj.view(cls)