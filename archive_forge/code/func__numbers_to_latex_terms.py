from typing import List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError
def _numbers_to_latex_terms(numbers: List[complex], decimals: int=10) -> List[str]:
    """Convert a list of numbers to latex formatted terms

    The first non-zero term is treated differently. For this term a leading + is suppressed.

    Args:
        numbers: List of numbers to format
        decimals: Number of decimal places to round to (default: 10).
    Returns:
        List of formatted terms
    """
    first_term = True
    terms = []
    for number in numbers:
        term = _num_to_latex(number, decimals=decimals, first_term=first_term, coefficient=True)
        terms.append(term)
        first_term = False
    return terms