import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def __verify_train_val_loop_configuration(trainer: 'pl.Trainer', model: 'pl.LightningModule') -> None:
    has_training_step = is_overridden('training_step', model)
    if not has_training_step:
        raise MisconfigurationException('No `training_step()` method defined. Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.')
    has_optimizers = is_overridden('configure_optimizers', model)
    if not has_optimizers:
        raise MisconfigurationException('No `configure_optimizers()` method defined. Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.')
    has_val_loader = trainer.fit_loop.epoch_loop.val_loop._data_source.is_defined()
    has_val_step = is_overridden('validation_step', model)
    if has_val_loader and (not has_val_step):
        rank_zero_warn('You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.')
    if has_val_step and (not has_val_loader):
        rank_zero_warn('You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.', category=PossibleUserWarning)
    if callable(getattr(model, 'training_epoch_end', None)):
        raise NotImplementedError(f'Support for `training_epoch_end` has been removed in v2.0.0. `{type(model).__name__}` implements this method. You can use the `on_train_epoch_end` hook instead. To access outputs, save them in-memory as instance attributes. You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520.')
    if callable(getattr(model, 'validation_epoch_end', None)):
        raise NotImplementedError(f'Support for `validation_epoch_end` has been removed in v2.0.0. `{type(model).__name__}` implements this method. You can use the `on_validation_epoch_end` hook instead. To access outputs, save them in-memory as instance attributes. You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520.')