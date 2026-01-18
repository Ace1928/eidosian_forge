import ast
from typing import Callable, Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from .classical_element import ClassicalElement
from .classical_function_visitor import ClassicalFunctionVisitor
from .utils import tweedledum2qiskit
The list of qregs used by the classicalfunction