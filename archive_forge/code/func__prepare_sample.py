import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import torch
import torch.nn as nn
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
def _prepare_sample(self, seq_group_metadata_list: List[SequenceGroupMetadata], prompt_lens: List[int], subquery_lens: Optional[List[int]]) -> SamplingMetadata:
    seq_groups: List[Tuple[List[int], SamplingParams]] = []
    selected_token_indices: List[int] = []
    generators: List[torch.Generator] = []
    selected_token_start_idx = 0
    categorized_sample_indices = {t: [] for t in SamplingType}
    categorized_sample_indices_start_idx = 0
    pin_memory = not self.in_wsl and (not self.device_config.is_neuron)
    max_subquery_len = max(subquery_lens) if subquery_lens else 1
    for i, seq_group_metadata in enumerate(seq_group_metadata_list):
        seq_ids = list(seq_group_metadata.seq_data.keys())
        sampling_params = seq_group_metadata.sampling_params
        seq_groups.append((seq_ids, sampling_params))
        if seq_group_metadata.is_prompt:
            assert len(seq_ids) == 1
            assert subquery_lens is not None
            subquery_len = subquery_lens[i]
            if sampling_params.prompt_logprobs is not None:
                categorized_sample_indices_start_idx += subquery_len - 1
            categorized_sample_indices[sampling_params.sampling_type].append(categorized_sample_indices_start_idx)
            categorized_sample_indices_start_idx += 1
            if sampling_params.prompt_logprobs is not None:
                selected_token_indices.extend(range(selected_token_start_idx, selected_token_start_idx + subquery_len - 1))
            selected_token_indices.append(selected_token_start_idx + subquery_len - 1)
            selected_token_start_idx += max_subquery_len
            if sampling_params.seed is not None:
                seq_group_metadata.state.generator = torch.Generator(device='cuda').manual_seed(sampling_params.seed)
        else:
            num_seqs = len(seq_ids)
            selected_token_indices.extend(range(selected_token_start_idx, selected_token_start_idx + num_seqs))
            selected_token_start_idx += num_seqs
            categorized_sample_indices[sampling_params.sampling_type].extend(range(categorized_sample_indices_start_idx, categorized_sample_indices_start_idx + num_seqs))
            categorized_sample_indices_start_idx += num_seqs
        if sampling_params.seed is not None:
            generators.append(seq_group_metadata.state.generator)
    selected_token_indices = _async_h2d(selected_token_indices, dtype=torch.long, target_device=self.device, pin_memory=pin_memory)
    categorized_sample_indices = {t: _async_h2d(seq_ids, dtype=torch.int, target_device=self.device, pin_memory=pin_memory) for t, seq_ids in categorized_sample_indices.items()}
    seq_data: Dict[int, SequenceData] = {}
    for seq_group_metadata in seq_group_metadata_list:
        seq_data.update(seq_group_metadata.seq_data)
    sampling_metadata = SamplingMetadata(seq_groups=seq_groups, seq_data=seq_data, prompt_lens=prompt_lens, selected_token_indices=selected_token_indices, categorized_sample_indices=categorized_sample_indices, generators=generators)
    return sampling_metadata