import abc
import dataclasses
from typing import Mapping, Tuple, TYPE_CHECKING, FrozenSet
import sympy
from cirq._compat import proper_repr
from cirq.protocols import json_serialization, measurement_key_protocol as mkp
from cirq.value import measurement_key
@dataclasses.dataclass(frozen=True)
class KeyCondition(Condition):
    """A classical control condition based on a single measurement key.

    This condition resolves to True iff the measurement key is non-zero at the
    time of resolution.
    """
    key: 'cirq.MeasurementKey'
    index: int = -1

    @property
    def keys(self):
        return (self.key,)

    def replace_key(self, current: 'cirq.MeasurementKey', replacement: 'cirq.MeasurementKey'):
        return KeyCondition(replacement) if self.key == current else self

    def __str__(self):
        return str(self.key) if self.index == -1 else f'{self.key}[{self.index}]'

    def __repr__(self):
        if self.index != -1:
            return f'cirq.KeyCondition({self.key!r}, {self.index})'
        return f'cirq.KeyCondition({self.key!r})'

    def resolve(self, classical_data: 'cirq.ClassicalDataStoreReader') -> bool:
        if self.key not in classical_data.keys():
            raise ValueError(f'Measurement key {self.key} missing when testing classical control')
        return classical_data.get_int(self.key, self.index) != 0

    def _json_dict_(self):
        return json_serialization.dataclass_json_dict(self)

    @classmethod
    def _from_json_dict_(cls, key, **kwargs):
        return cls(key=key)

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')