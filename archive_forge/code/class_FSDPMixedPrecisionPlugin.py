import sys
from typing import TYPE_CHECKING, Any, Literal, Optional
import pytorch_lightning as pl
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation
from pytorch_lightning.plugins.precision import (
class FSDPMixedPrecisionPlugin(FSDPPrecision):
    """AMP for Fully Sharded Data Parallel (FSDP) Training.

    .. deprecated:: Use :class:`FSDPPrecision` instead.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self, precision: Literal['16-mixed', 'bf16-mixed'], device: str, scaler: Optional['ShardedGradScaler']=None) -> None:
        rank_zero_deprecation(f'The `{type(self).__name__}` is deprecated. Use `pytorch_lightning.plugins.precision.FSDPPrecision` instead.')
        super().__init__(precision=precision, scaler=scaler)