import sys
from typing import Any
import pytorch_lightning as pl
from lightning_fabric.strategies import _StrategyRegistry
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.plugins.precision import XLAPrecision
from pytorch_lightning.strategies.single_xla import SingleDeviceXLAStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation
class XLABf16PrecisionPlugin(XLAPrecision):
    """Legacy class.

    Use :class:`~pytorch_lightning.plugins.precision.xlabf16.XLAPrecision` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation('The `XLABf16PrecisionPlugin` class is deprecated. Use `pytorch_lightning.plugins.precision.XLAPrecision` instead.')
        super().__init__(precision='bf16-true')