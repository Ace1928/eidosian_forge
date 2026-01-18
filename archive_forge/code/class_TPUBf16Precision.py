import sys
from typing import Any
import lightning_fabric as fabric
from lightning_fabric.accelerators import XLAAccelerator
from lightning_fabric.plugins.precision import XLAPrecision
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.single_xla import SingleDeviceXLAStrategy
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation
class TPUBf16Precision(XLABf16Precision):
    """Legacy class.

    Use :class:`~lightning_fabric.plugins.precision.xla.XLAPrecision` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation('The `TPUBf16Precision` class is deprecated. Use `lightning_fabric.plugins.precision.XLAPrecision` instead.')
        super().__init__(*args, **kwargs)