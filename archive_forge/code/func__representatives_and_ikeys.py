from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence
def _representatives_and_ikeys(self, point):
    return [(rep, ikey) for rep in self.representatives(point) for ikey in floor_as_integers(self._RF_scale * self.float_hash(rep))]