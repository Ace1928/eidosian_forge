from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
@dataclass
class DiagramSettings:
    """
    Settings to control the layout and rendering of circuits.
    """
    texify_numerical_constants: bool = True
    '\n    Convert numerical constants, such as pi, to LaTeX form.\n    '
    impute_missing_qubits: bool = False
    '\n    Include qubits with indices between those explicitly referenced in the Quil program.\n\n    For example, if true, the diagram for `CNOT 0 2` would have three qubit lines: 0, 1, 2.\n    '
    label_qubit_lines: bool = True
    '\n    Label qubit lines.\n    '
    abbreviate_controlled_rotations: bool = False
    '\n    Write controlled rotations in a compact form.\n\n    For example,  `RX(pi)` as `X_{\\pi}`, instead of the longer `R_X(\\pi)`\n    '
    qubit_line_open_wire_length: int = 1
    '\n    The length by which qubit lines should be extended with open wires at the right of the diagram.\n\n    The default of 1 is the natural choice. The main reason for including this option\n    is that it may be appropriate for this to be 0 in subdiagrams.\n    '
    right_align_terminal_measurements: bool = True
    '\n    Align measurement operations which appear at the end of the program.\n    '