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
from ..utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig
from .logits_process import (
from .stopping_criteria import (
def _speculative_sampling(candidate_input_ids, candidate_logits, candidate_length, new_logits, last_assistant_token_is_eos, max_matches):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()
    if last_assistant_token_is_eos and n_matches == candidate_length:
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, :n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)
        gamma = min(candidate_logits.shape[1], max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp(p_n_plus_1 - q_n_plus_1, min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t
    return (valid_tokens, n_matches)