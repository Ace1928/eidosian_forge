from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def TIKZ_GATE_GROUP(qubits: Sequence[int], width: int, label: str) -> str:
    num_qubits = max(qubits) - min(qubits) + 1
    return '\\gategroup[{qubits},steps={width},style={{dashed, rounded corners,fill=blue!20, inner xsep=2pt}}, background]{{{label}}}'.format(qubits=num_qubits, width=width, label=label)