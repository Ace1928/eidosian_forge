from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def _H_decompose(self, v, y, z, delta):
    """Determines the transformation

                H^v (|y> + i^delta |z>) = omega S^a H^b |c>

        where the state represents a single qubit.

        Input: v,y,z are boolean; delta is an integer (mod 4)
        Outputs: a,b,c are boolean; omega is a complex number

        Precondition: y != z"""
    if y == z:
        raise ValueError('|y> is equal to |z>')
    if not v:
        omega = 1j ** (delta * int(y))
        delta2 = (-1) ** y * delta % 4
        c = bool(delta2 >> 1)
        a = bool(delta2 & 1)
        b = True
    elif not delta & 1:
        a = False
        b = False
        c = bool(delta >> 1)
        omega = (-1) ** (c & y)
    else:
        omega = 1 / np.sqrt(2) * (1 + 1j ** delta)
        b = True
        a = True
        c = not delta >> 1 ^ y
    return (omega, a, b, c)