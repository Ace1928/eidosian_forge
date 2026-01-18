import os
import queue
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import torch.multiprocessing as mp
from typing_extensions import override
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.strategies.launchers.xla import _rank_teardown
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.strategies.launchers.multiprocessing import (
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
@override
def _collect_rank_zero_results(self, trainer: 'pl.Trainer', results: Any) -> Optional['_WorkerOutput']:
    rank_zero_debug('Collecting results from rank 0 process.')
    checkpoint_callback = trainer.checkpoint_callback
    best_model_path = checkpoint_callback.best_model_path if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') else None
    weights_path = None
    if trainer.state.fn == TrainerFn.FITTING:
        state_dict = self._strategy.lightning_module_state_dict()
        weights_path = os.path.join(trainer.default_root_dir, '.temp.ckpt')
        self._strategy.checkpoint_io.save_checkpoint(state_dict, weights_path)
    if self._strategy.local_rank != 0:
        return None
    extra = self.get_extra_results(trainer)
    return _WorkerOutput(best_model_path, weights_path, trainer.state, results, extra)