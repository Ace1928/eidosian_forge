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
def _validate_observe_invariants(self):
    """
        Check that we properly called self_observe after the last batch_act.
        """
    if self.__expecting_to_reply:
        raise RuntimeError('Last observe() had a label, but no call to self_observe ever happened. You are likely making multiple observe() calls without a corresponding act(). This was changed in #2043. File a GitHub issue if you require assistance.')
    if self.__expecting_clear_history:
        raise RuntimeError('Last observe() was episode_done, but we never saw a corresponding self_observe to clear the history, probably because you missed an act(). This was changed in #2043. File a GitHub issue if you require assistance.')