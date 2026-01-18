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
def _prepare_prompt(self, seq_group_metadata_list: List[SequenceGroupMetadata]) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int], List[int], List[int], Set[LoRARequest]]:
    assert len(seq_group_metadata_list) > 0
    input_tokens: List[List[int]] = []
    input_positions: List[List[int]] = []
    slot_mapping: List[List[int]] = []
    lora_index_mapping: List[int] = []
    lora_prompt_mapping: List[int] = []
    lora_requests: Set[LoRARequest] = set()
    prompt_lens: List[int] = []
    context_lens: List[int] = []
    subquery_lens: List[int] = []
    prefix_block_tables: List[List[int]] = []
    for seq_group_metadata in seq_group_metadata_list:
        assert seq_group_metadata.is_prompt
        seq_ids = list(seq_group_metadata.seq_data.keys())
        assert len(seq_ids) == 1
        seq_id = seq_ids[0]
        seq_data = seq_group_metadata.seq_data[seq_id]
        prompt_tokens = seq_data.get_token_ids()
        prompt_len = len(prompt_tokens)
        prompt_lens.append(prompt_len)
        prefix_len = 0
        prefix = seq_group_metadata.prefix
        if prefix is not None and prefix.computed:
            prefix_len = prefix.get_length()
            prompt_tokens = prompt_tokens[prefix_len:]
            prefix_block_tables.append(prefix.get_block_numbers())
        else:
            prefix_block_tables.append([])
        context_lens.append(prefix_len)
        subquery_lens.append(prompt_len - prefix_len)
        input_tokens.append(prompt_tokens)
        input_positions.append(list(range(prefix_len, prefix_len + len(prompt_tokens))))
        lora_id = seq_group_metadata.lora_int_id
        if lora_id > 0:
            lora_requests.add(seq_group_metadata.lora_request)
        lora_index_mapping.append([lora_id] * (prompt_len - prefix_len))
        lora_prompt_mapping.extend([lora_id] * (prompt_len - prefix_len if seq_group_metadata.sampling_params.prompt_logprobs else 1))
        if seq_group_metadata.block_tables is None:
            slot_mapping.append([_PAD_SLOT_ID] * prompt_len)
            continue
        slot_mapping.append([])
        block_table = seq_group_metadata.block_tables[seq_id]
        start_idx = 0
        if self.sliding_window is not None:
            assert prefix_len == 0, 'Prefix caching is currently not supported with sliding window attention'
            start_idx = max(0, prompt_len - self.sliding_window)
        for i in range(prefix_len, prompt_len):
            if i < start_idx:
                slot_mapping[-1].append(_PAD_SLOT_ID)
                continue
            block_number = block_table[i // self.block_size]
            block_offset = i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping[-1].append(slot)
    max_prompt_len = max(subquery_lens)
    input_tokens = _make_tensor_with_pad(input_tokens, max_prompt_len, pad=0, dtype=torch.long, device=self.device)
    input_positions = _make_tensor_with_pad(input_positions, max_prompt_len, pad=0, dtype=torch.long, device=self.device)
    slot_mapping = _make_tensor_with_pad(slot_mapping, max_prompt_len, pad=_PAD_SLOT_ID, dtype=torch.long, device=self.device)
    lora_index_mapping = [_pad_to_max(mapping, max_prompt_len, pad=0) for mapping in lora_index_mapping]
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.int, device=self.device)
    max_prompt_block_table_len = max((len(t) for t in prefix_block_tables))
    block_tables = _make_tensor_with_pad(prefix_block_tables, max_len=max_prompt_block_table_len, pad=0, dtype=torch.int, device=self.device)
    start_loc_tensor = torch.arange(0, len(prompt_lens) * max_prompt_len, max_prompt_len, dtype=torch.long, device=self.device)
    prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
    input_metadata = InputMetadata(is_prompt=True, slot_mapping=slot_mapping, prompt_lens=prompt_lens_tensor, max_seq_len=max_prompt_len, start_loc=start_loc_tensor, max_context_len=None, context_lens=context_lens_tensor, block_tables=block_tables, use_cuda_graph=False, kv_cache_dtype=self.kv_cache_dtype)
    return (input_tokens, input_positions, input_metadata, prompt_lens, subquery_lens, lora_index_mapping, lora_prompt_mapping, lora_requests)