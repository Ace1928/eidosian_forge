from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _apply_top_k_top_p(logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    top_k_mask = logits_sort.size(1) - k.to(torch.long)
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float('inf'))
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float('inf'))
    src = torch.arange(logits_idx.shape[-1], device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1, index=logits_idx, src=src)
    logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
    return logits