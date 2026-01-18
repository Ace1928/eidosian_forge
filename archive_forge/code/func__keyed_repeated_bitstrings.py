import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _keyed_repeated_bitstrings(vals: Mapping[str, np.ndarray]) -> str:
    keyed_bitstrings = []
    for key in sorted(vals.keys()):
        reps = vals[key]
        n = 0 if len(reps) == 0 else len(reps[0])
        all_bits = ', '.join((_bitstring(reps[:, i]) for i in range(n)))
        keyed_bitstrings.append(f'{key}={all_bits}')
    return '\n'.join(keyed_bitstrings)