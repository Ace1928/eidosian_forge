from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
def _validate_self_observe_invariants(self):
    """
        Check some invariant conditions for self_observe.

        Goal is to catch potential places where we forget to call self_observe.
        """
    if self.observation is None:
        raise RuntimeError("You're self_observing without having observed something. Check if you're missing a step in your observe/act/self_observe loop.")
    if self.observation['episode_done']:
        if not self.__expecting_clear_history:
            raise RuntimeError('You probably overrode observe() without implementing calling super().observe(). This is unexpected. *If you must* avoid the super call, then you should file a GitHub issue referencing #2043.')