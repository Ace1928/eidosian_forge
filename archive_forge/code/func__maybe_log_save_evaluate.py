import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_utils import EvalLoopOutput
from ..core import PPODecorators
from ..import_utils import is_peft_available
def _maybe_log_save_evaluate(self):
    if self.args.eval_steps is not None:
        if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step != 0:
            self.evaluate(self.eval_dataset)
    if self.args.logging_steps is not None:
        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step != 0:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(self.tr_loss).mean().item()
            self.tr_loss -= self.tr_loss
            logs['loss'] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs['learning_rate'] = self._get_learning_rate()
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)