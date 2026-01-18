import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def __verify_batch_transfer_support(trainer: 'pl.Trainer') -> None:
    batch_transfer_hooks = ('transfer_batch_to_device', 'on_after_batch_transfer')
    datahook_selector = trainer._data_connector._datahook_selector
    assert datahook_selector is not None
    for hook in batch_transfer_hooks:
        if _graphcore_available_and_importable():
            from lightning_graphcore import IPUAccelerator
            if isinstance(trainer.accelerator, IPUAccelerator) and (is_overridden(hook, datahook_selector.model) or is_overridden(hook, datahook_selector.datamodule)):
                raise MisconfigurationException(f'Overriding `{hook}` is not supported with IPUs.')