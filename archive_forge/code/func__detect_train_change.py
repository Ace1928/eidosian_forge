from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd.profiler as profiler
import torch.distributed as dist
from fairscale.internal.params import Workhandle, get_global_rank
from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
def _detect_train_change(self) -> bool:
    with profiler.record_function('fairscale::sdp::detect_train_changes'):
        trainable_mask = list(map(_trainable, self._all_params))
        trainability_changed = trainable_mask != self._reference_trainable_mask
        trainability_changed |= not self.training and len(self._grad_hooks) > 0
        if self._warn_on_trainable_params_changed and trainability_changed:
            logging.warning('ShardedDDP detected that the trainable params changed, either because of eval/train mode or parameter freezing/unfreeze.')
            self._reference_trainable_mask = trainable_mask
    return trainability_changed