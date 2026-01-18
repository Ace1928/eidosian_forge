from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor, output_tokens_tensor: torch.Tensor, presence_penalties: torch.Tensor, frequency_penalties: torch.Tensor, repetition_penalties: torch.Tensor) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _get_bin_counts_and_mask(prompt_tokens_tensor, vocab_size, num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(output_tokens_tensor, vocab_size, num_seqs)
    repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
    repetition_penalties[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties, logits * repetition_penalties)
    logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits