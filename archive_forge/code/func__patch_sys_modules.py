import sys
from typing import Any
import lightning_fabric as fabric
from lightning_fabric.accelerators import XLAAccelerator
from lightning_fabric.plugins.precision import XLAPrecision
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.single_xla import SingleDeviceXLAStrategy
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation
def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules['lightning_fabric.strategies.single_tpu'] = self
    sys.modules['lightning_fabric.accelerators.tpu'] = self
    sys.modules['lightning_fabric.plugins.precision.tpu'] = self
    sys.modules['lightning_fabric.plugins.precision.tpu_bf16'] = self
    sys.modules['lightning_fabric.plugins.precision.xlabf16'] = self