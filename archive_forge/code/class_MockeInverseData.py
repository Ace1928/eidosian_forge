from dataclasses import dataclass
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor
@dataclass
class MockeInverseData:
    var_offsets: dict
    x_length: int
    var_shapes: dict
    param_shapes: dict
    param_to_size: dict
    param_id_map: dict