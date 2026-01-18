import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def __warn_dataloader_iter_limitations(model: 'pl.LightningModule') -> None:
    """Check if `dataloader_iter is enabled`."""
    if any((is_param_in_hook_signature(step_fn, 'dataloader_iter', explicit=True) for step_fn in (model.training_step, model.validation_step, model.predict_step, model.test_step) if step_fn is not None)):
        rank_zero_warn('You are using the `dataloader_iter` step flavor. If you consume the iterator more than once per step, the `batch_idx` argument in any hook that takes it will not match with the batch index of the last batch consumed. This might have unforeseen effects on callbacks or code that expects to get the correct index. This will also not work well with gradient accumulation. This feature is very experimental and subject to change. Here be dragons.', category=PossibleUserWarning)