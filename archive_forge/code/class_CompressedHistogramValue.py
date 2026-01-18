import dataclasses
import numpy as np
from typing import Tuple
@dataclasses.dataclass(frozen=True)
class CompressedHistogramValue:
    """Represents a value in a compressed histogram.

    Attributes:
      basis_point: Compression point represented in basis point, 1/100th of a
        percent.
      value: Cumulative weight at the basis point.
    """
    basis_point: float
    value: float

    def as_tuple(self) -> Tuple[float, float]:
        """Returns the basis point and the value as a tuple."""
        return (self.basis_point, self.value)