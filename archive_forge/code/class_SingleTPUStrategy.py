import sys
from typing import Any
import lightning_fabric as fabric
from lightning_fabric.accelerators import XLAAccelerator
from lightning_fabric.plugins.precision import XLAPrecision
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.single_xla import SingleDeviceXLAStrategy
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation
class SingleTPUStrategy(SingleDeviceXLAStrategy):
    """Legacy class.

    Use :class:`~lightning_fabric.strategies.single_xla.SingleDeviceXLAStrategy` instead.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation("The 'single_tpu' strategy is deprecated. Use 'single_xla' instead.")
        super().__init__(*args, **kwargs)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if 'single_tpu' not in strategy_registry:
            strategy_registry.register('single_tpu', cls, description='Legacy class. Use `single_xla` instead.')