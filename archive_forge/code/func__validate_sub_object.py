from typing import (
import numpy as np
from cirq import protocols, value, _import
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _validate_sub_object(sub_object: Union['cirq.Gate', 'cirq.Operation']):
    if protocols.is_measurement(sub_object):
        raise ValueError(f'Cannot control measurement {sub_object}')
    if not protocols.has_mixture(sub_object) and (not protocols.is_parameterized(sub_object)):
        raise ValueError(f'Cannot control channel with non-unitary operators: {sub_object}')