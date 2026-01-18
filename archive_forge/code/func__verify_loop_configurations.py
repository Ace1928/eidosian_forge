import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def _verify_loop_configurations(trainer: 'pl.Trainer') -> None:
    """Checks that the model is configured correctly before the run is started.

    Args:
        trainer: Lightning Trainer. Its `lightning_module` (the model) to check the configuration.

    """
    model = trainer.lightning_module
    if trainer.state.fn is None:
        raise ValueError('Unexpected: Trainer state fn must be set before validating loop configuration.')
    if trainer.state.fn == TrainerFn.FITTING:
        __verify_train_val_loop_configuration(trainer, model)
        __verify_manual_optimization_support(trainer, model)
    elif trainer.state.fn == TrainerFn.VALIDATING:
        __verify_eval_loop_configuration(model, 'val')
    elif trainer.state.fn == TrainerFn.TESTING:
        __verify_eval_loop_configuration(model, 'test')
    elif trainer.state.fn == TrainerFn.PREDICTING:
        __verify_eval_loop_configuration(model, 'predict')
    __verify_batch_transfer_support(trainer)
    __verify_configure_model_configuration(model)
    __warn_dataloader_iter_limitations(model)