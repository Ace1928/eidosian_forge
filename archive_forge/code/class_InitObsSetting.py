import dataclasses
import numbers
from typing import (
import sympy
from cirq import ops, value, protocols
@dataclasses.dataclass(frozen=True)
class InitObsSetting:
    """A pair of initial state and observable.

    Usually, given a circuit you want to iterate through many
    InitObsSettings to vary the initial state preparation and output
    observable.
    """
    init_state: value.ProductState
    observable: ops.PauliString

    def __post_init__(self):
        init_qs = self.init_state.qubits
        obs_qs = self.observable.qubits
        if set(obs_qs) > set(init_qs):
            raise ValueError(f"`observable`'s qubits should be a subset of those found in `init_state`. observable qubits: {obs_qs}. init_state qubits: {init_qs}")

    def __str__(self):
        return f'{self.init_state} → {self.observable}'

    def __repr__(self):
        return f'cirq.work.InitObsSetting(init_state={self.init_state!r}, observable={self.observable!r})'

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)