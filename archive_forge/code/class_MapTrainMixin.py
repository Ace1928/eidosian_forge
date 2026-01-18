from __future__ import annotations
import typing
from .._utils.registry import alias
from ..doctools import document
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
class MapTrainMixin:
    """
    Override map and train methods
    """
    guide = None

    def map(self, x, limits=None) -> Sequence[Any]:
        """
        Identity map

        Notes
        -----
        Identity scales bypass the palette completely since the
        map is the identity function.
        """
        return x

    def train(self, x, drop=False):
        if self.guide is None:
            return
        return super().train(x)