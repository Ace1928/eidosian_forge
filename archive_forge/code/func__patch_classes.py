import sys
from typing import Any
import lightning_fabric as fabric
from lightning_fabric.accelerators import XLAAccelerator
from lightning_fabric.plugins.precision import XLAPrecision
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.single_xla import SingleDeviceXLAStrategy
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation
def _patch_classes() -> None:
    setattr(fabric.strategies, 'SingleTPUStrategy', SingleTPUStrategy)
    setattr(fabric.accelerators, 'TPUAccelerator', TPUAccelerator)
    setattr(fabric.plugins, 'TPUPrecision', TPUPrecision)
    setattr(fabric.plugins.precision, 'TPUPrecision', TPUPrecision)
    setattr(fabric.plugins, 'TPUBf16Precision', TPUBf16Precision)
    setattr(fabric.plugins.precision, 'TPUBf16Precision', TPUBf16Precision)
    setattr(fabric.plugins, 'XLABf16Precision', XLABf16Precision)
    setattr(fabric.plugins.precision, 'XLABf16Precision', XLABf16Precision)