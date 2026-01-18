import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def __verify_manual_optimization_support(trainer: 'pl.Trainer', model: 'pl.LightningModule') -> None:
    if model.automatic_optimization:
        return
    if trainer.gradient_clip_val is not None and trainer.gradient_clip_val > 0:
        raise MisconfigurationException(f'Automatic gradient clipping is not supported for manual optimization. Remove `Trainer(gradient_clip_val={trainer.gradient_clip_val})` or switch to automatic optimization.')
    if trainer.accumulate_grad_batches != 1:
        raise MisconfigurationException(f'Automatic gradient accumulation is not supported for manual optimization. Remove `Trainer(accumulate_grad_batches={trainer.accumulate_grad_batches})` or switch to automatic optimization.')