from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def _col_op(mat, ctrl, trgt):
    mat[:, ctrl] = mat[:, trgt] ^ mat[:, ctrl]