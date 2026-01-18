from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def _row_op(mat, ctrl, trgt):
    mat[trgt] = mat[trgt] ^ mat[ctrl]