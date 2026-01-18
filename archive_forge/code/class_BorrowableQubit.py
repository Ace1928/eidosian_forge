import abc
import dataclasses
from typing import Iterable, List, TYPE_CHECKING
from cirq.ops import raw_types
class BorrowableQubit(_BaseAncillaQid):
    """An internal qid type that represents a dirty ancilla allocation."""

    def __str__(self) -> str:
        dim_str = f' (d={self.dimension})' if self.dim != 2 else ''
        return f'{self.prefix}_b({self.id}){dim_str}'