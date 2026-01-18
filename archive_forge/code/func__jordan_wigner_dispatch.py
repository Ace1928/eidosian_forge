from functools import singledispatch
from typing import Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord
from .fermionic import FermiSentence, FermiWord
@singledispatch
def _jordan_wigner_dispatch(fermi_operator, ps, wire_map, tol):
    """Dispatches to appropriate function if fermi_operator is a FermiWord or FermiSentence."""
    raise ValueError(f'fermi_operator must be a FermiWord or FermiSentence, got: {fermi_operator}')