import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
def _extract_qubit_index(qubit: Union[Qubit, QubitPlaceholder, FormalArgument], index: bool=True) -> QubitDesignator:
    if index and isinstance(qubit, Qubit):
        return qubit.index
    return qubit