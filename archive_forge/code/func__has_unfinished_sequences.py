import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
from .stopping_criteria import (
def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
    """
        Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
        fed through `this_peer_finished`. ZeRO stage 3-friendly.
        """
    if synced_gpus:
        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
        dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
        if this_peer_finished_flag.item() == 0.0:
            return False
    elif this_peer_finished:
        return False
    return True