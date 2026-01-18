from __future__ import annotations
import typing
from mizani.scale import scale_continuous, scale_discrete
class RangeContinuous(Range):
    """
    Continuous Range
    """
    range: TupleFloat2

    def train(self, x: FloatArrayLike):
        """
        Train continuous range
        """
        rng = None if self.is_empty() else self.range
        self.range = scale_continuous.train(x, rng)