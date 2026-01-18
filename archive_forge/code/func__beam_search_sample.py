from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _beam_search_sample(selected_seq_groups: List[Tuple[List[int], SamplingParams]], is_prompts: List[bool], seq_data: Dict[int, SequenceData], logprobs: torch.Tensor) -> List[Tuple[List[int], List[int]]]:
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        beam_width = sampling_params.best_of
        seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
        if is_prompt:
            assert num_parent_seqs == 1, 'Prompt input should have only one seq.'
            parent_ids = [0] * (2 * beam_width)
            _, next_token_ids = torch.topk(seq_group_logprobs[0], 2 * beam_width)
            next_token_ids = next_token_ids.tolist()
        else:
            cumulative_logprobs = [seq_data[seq_id].cumulative_logprob for seq_id in seq_ids]
            cumulative_logprobs = torch.tensor(cumulative_logprobs, dtype=torch.float, device=seq_group_logprobs.device)
            seq_group_logprobs = seq_group_logprobs + cumulative_logprobs.unsqueeze(dim=1)
            _, topk_ids = torch.topk(seq_group_logprobs.flatten(), 2 * beam_width)
            topk_ids = topk_ids.tolist()
            vocab_size = seq_group_logprobs.size(-1)
            parent_ids = [i // vocab_size for i in topk_ids]
            next_token_ids = [i % vocab_size for i in topk_ids]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    return results