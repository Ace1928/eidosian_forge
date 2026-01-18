from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
class _Unit(Sweep):
    """A sweep with a single element that assigns no parameter values.

    This is useful as a base sweep, instead of special casing None.
    """

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return True

    @property
    def keys(self) -> List['cirq.TParamKey']:
        return []

    def __len__(self) -> int:
        return 1

    def param_tuples(self) -> Iterator[Params]:
        yield ()

    def __repr__(self) -> str:
        return 'cirq.UnitSweep'

    def _json_dict_(self) -> Dict[str, Any]:
        return {}