from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _random_sample(selected_seq_groups: List[Tuple[List[int], SamplingParams]], is_prompts: List[bool], random_samples: torch.Tensor) -> List[Tuple[List[int], List[int]]]:
    random_samples = random_samples.cpu()
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = random_samples[sample_idx, :sampling_params.best_of].tolist()
        else:
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx + num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results