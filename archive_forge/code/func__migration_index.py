import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migration_index() -> Dict[str, List[Callable[[_CHECKPOINT], _CHECKPOINT]]]:
    """Migration functions returned here will get executed in the order they are listed."""
    return {'0.10.0': [_migrate_model_checkpoint_early_stopping], '1.6.0': [_migrate_loop_global_step_to_progress_tracking, _migrate_loop_current_epoch_to_progress_tracking], '1.6.5': [_migrate_loop_batches_that_stepped], '1.9.0': [_migrate_model_checkpoint_save_on_train_epoch_end_default], '2.0.0': [_drop_apex_amp_state, _migrate_loop_structure_after_tbptt_removal, _migrate_loop_structure_after_optimizer_loop_removal, _migrate_loop_structure_after_dataloader_loop_removal]}