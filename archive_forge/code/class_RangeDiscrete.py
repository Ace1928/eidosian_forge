from __future__ import annotations
import typing
from mizani.scale import scale_continuous, scale_discrete
class RangeDiscrete(Range):
    """
    Discrete Range
    """
    range: Sequence[Any]

    def train(self, x: AnyArrayLike, drop: bool=False, na_rm: bool=False):
        """
        Train discrete range
        """
        rng = None if self.is_empty() else self.range
        self.range = scale_discrete.train(x, rng, drop, na_rm=na_rm)