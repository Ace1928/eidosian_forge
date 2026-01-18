from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.utils import in_wsl, is_neuron
class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        selected_token_indices: Token indices selected for sampling.
        categorized_sample_indices: SamplingType -> token indices to sample.
        generators: List of torch.Generators to use for seeded sampling
        perform_sampling: Whether to perform sampling. This option is used to
            make the sampling only happens in the driver worker, and disable
            sampling in other worker processes.
    """

    def __init__(self, seq_groups: Optional[List[Tuple[List[int], SamplingParams]]], seq_data: Optional[Dict[int, SequenceData]], prompt_lens: Optional[List[int]], selected_token_indices: torch.Tensor, categorized_sample_indices: Optional[Dict[SamplingType, torch.Tensor]], generators: Optional[List[torch.Generator]]=None, perform_sampling: bool=True) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.generators = generators
        self.perform_sampling = perform_sampling
        self.num_prompts = len(prompt_lens) if prompt_lens is not None else 0

    def __repr__(self) -> str:
        return f'SamplingMetadata(seq_groups={self.seq_groups}, seq_data={self.seq_data}, prompt_lens={self.prompt_lens}, selected_token_indices={self.selected_token_indices}, categorized_sample_indices={self.categorized_sample_indices}), perform_sampling={self.perform_sampling})'