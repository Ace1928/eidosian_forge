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
def _update_learning_rates(self, interval: str, update_plateau_schedulers: bool) -> None:
    """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.

        """
    trainer = self.trainer
    if not trainer.lr_scheduler_configs or not trainer.lightning_module.automatic_optimization:
        return
    for config in trainer.lr_scheduler_configs:
        if update_plateau_schedulers ^ config.reduce_on_plateau:
            continue
        current_idx = self.batch_idx if interval == 'step' else trainer.current_epoch
        current_idx += 1
        if config.interval == interval and current_idx % config.frequency == 0:
            monitor_val = None
            if config.reduce_on_plateau:
                monitor_key = config.monitor
                assert monitor_key is not None
                monitor_val = self._get_monitor_value(monitor_key)
                if monitor_val is None:
                    if config.strict:
                        avail_metrics = list(trainer.callback_metrics)
                        raise MisconfigurationException(f'ReduceLROnPlateau conditioned on metric {monitor_key} which is not available. Available metrics are: {avail_metrics}. Condition can be set using `monitor` key in lr scheduler dict')
                    rank_zero_warn(f'ReduceLROnPlateau conditioned on metric {monitor_key} which is not available but strict is set to `False`. Skipping learning rate update.', category=RuntimeWarning)
                    continue
            self.scheduler_progress.increment_ready()
            call._call_lightning_module_hook(trainer, 'lr_scheduler_step', config.scheduler, monitor_val)
            self.scheduler_progress.increment_completed()