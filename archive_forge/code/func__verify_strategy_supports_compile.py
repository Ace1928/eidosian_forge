from typing import Union
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy
from pytorch_lightning.utilities.model_helpers import _check_mixed_imports
def _verify_strategy_supports_compile(model: 'pl.LightningModule', strategy: Strategy) -> None:
    if model._compiler_ctx is not None:
        supported_strategies = (SingleDeviceStrategy, DDPStrategy, FSDPStrategy)
        if not isinstance(strategy, supported_strategies) or isinstance(strategy, DeepSpeedStrategy):
            supported_strategy_names = ', '.join((s.__name__ for s in supported_strategies))
            raise RuntimeError(f'Using a compiled model is incompatible with the current strategy: `{type(strategy).__name__}`. Only {supported_strategy_names} support compilation. Either switch to one of the supported strategies or avoid passing in compiled model.')