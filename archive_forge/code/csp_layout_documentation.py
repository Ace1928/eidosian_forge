import random
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils import optionals as _optionals
from qiskit.transpiler.target import Target
run the layout method