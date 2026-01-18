import abc
import dataclasses
from typing import Iterable, List, TYPE_CHECKING
from cirq.ops import raw_types
@dataclasses.dataclass(frozen=True)
class _BaseAncillaQid(raw_types.Qid):
    id: int
    dim: int = 2
    prefix: str = ''

    def _comparison_key(self) -> int:
        return self.id

    @property
    def dimension(self) -> int:
        return self.dim

    def __repr__(self) -> str:
        dim_str = f', dim={self.dim}' if self.dim != 2 else ''
        prefix_str = f', prefix={self.prefix}' if self.prefix != '' else ''
        return f'cirq.ops.{type(self).__name__}({self.id}{dim_str}{prefix_str})'