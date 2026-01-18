import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
def _get_frame_qubits(frame: Frame, index: bool=True) -> Set[QubitDesignator]:
    for q in frame.qubits:
        if isinstance(q, FormalArgument):
            raise ValueError('Attempted to extract FormalArgument where a Qubit is expected.')
    return {_extract_qubit_index(q, index) for q in cast(List[Qubit], frame.qubits)}