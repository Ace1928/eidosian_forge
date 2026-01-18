from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _apply_logits_processors(logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
    logits_row_idx = 0
    found_logits_processors = False
    for seq_ids, sampling_params in sampling_metadata.seq_groups:
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True
            for seq_id in seq_ids:
                logits_row = logits[logits_row_idx]
                token_ids = sampling_metadata.seq_data[seq_id].output_token_ids
                for logits_processor in logits_processors:
                    logits_row = logits_processor(token_ids, logits_row)
                logits[logits_row_idx] = logits_row
                logits_row_idx += 1
        else:
            logits_row_idx += len(seq_ids)
    if found_logits_processors:
        assert logits_row_idx == logits.shape[0]
    return logits