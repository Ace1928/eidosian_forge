from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _get_bin_counts_and_mask(tokens: torch.Tensor, vocab_size: int, num_seqs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_counts = torch.zeros((num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0
    return (bin_counts, mask)