from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def _first_differing_moment_index(circuit1: circuits.AbstractCircuit, circuit2: circuits.AbstractCircuit) -> Optional[int]:
    for i, (m1, m2) in enumerate(itertools.zip_longest(circuit1, circuit2)):
        if m1 != m2:
            return i
    return None