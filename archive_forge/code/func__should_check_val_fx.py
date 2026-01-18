import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _Stateful
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning import loops  # import as loops to avoid circular imports
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.optimization import _AutomaticOptimization, _ManualOptimization
from pytorch_lightning.loops.optimization.automatic import _OUTPUTS_TYPE as _OPTIMIZER_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.manual import _OUTPUTS_TYPE as _MANUAL_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.progress import _BatchProgress, _SchedulerProgress
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException, SIGTERMException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def _should_check_val_fx(self, data_fetcher: _DataFetcher) -> bool:
    """Decide if we should run validation."""
    if not self._should_check_val_epoch():
        return False
    is_infinite_dataset = self.trainer.val_check_batch == float('inf')
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)):
        return True
    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True
    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (self.batch_idx + 1) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float('inf'):
        current_iteration = self.total_batch_idx if self.trainer.check_val_every_n_epoch is None else self.batch_idx
        is_val_check_batch = (current_iteration + 1) % self.trainer.val_check_batch == 0
    return is_val_check_batch