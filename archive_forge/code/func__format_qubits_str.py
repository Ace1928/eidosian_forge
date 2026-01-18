import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
def _format_qubits_str(qubits: Iterable[Union[Qubit, QubitPlaceholder, FormalArgument]]) -> str:
    return ' '.join([_format_qubit_str(qubit) for qubit in qubits])