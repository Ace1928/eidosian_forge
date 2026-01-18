import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def __verify_eval_loop_configuration(model: 'pl.LightningModule', stage: str) -> None:
    step_name = 'validation_step' if stage == 'val' else f'{stage}_step'
    has_step = is_overridden(step_name, model)
    if stage == 'predict':
        if model.predict_step is None:
            raise MisconfigurationException('`predict_step` cannot be None to run `Trainer.predict`')
        if not has_step and (not is_overridden('forward', model)):
            raise MisconfigurationException('`Trainer.predict` requires `forward` method to run.')
    else:
        if not has_step:
            trainer_method = 'validate' if stage == 'val' else stage
            raise MisconfigurationException(f'No `{step_name}()` method defined to run `Trainer.{trainer_method}`.')
        epoch_end_name = 'validation_epoch_end' if stage == 'val' else 'test_epoch_end'
        if callable(getattr(model, epoch_end_name, None)):
            raise NotImplementedError(f'Support for `{epoch_end_name}` has been removed in v2.0.0. `{type(model).__name__}` implements this method. You can use the `on_{epoch_end_name}` hook instead. To access outputs, save them in-memory as instance attributes. You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520.')