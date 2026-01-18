import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
@staticmethod
def _prepare_decoder_input_ids(cur_bsz, init_tokens, current_segments, batch_idx_map, do_condition_on_prev_tokens, prompt_ids, generation_config, config, device, suppress_tokens, kwargs):
    cut_off_length = config.max_target_positions // 2 - 1
    one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
    decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)
    prev_start_of_text = getattr(generation_config, 'prev_sot_token_id', None)
    if prev_start_of_text is None:
        prev_start_of_text = suppress_tokens[-2] if suppress_tokens is not None else None
    if any(do_condition_on_prev_tokens) and len(current_segments[0]) > 0:
        active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]
        if prompt_ids is not None and generation_config.prompt_condition_type == 'all-segments':
            prev_ids = prompt_ids
        else:
            prev_ids = prev_start_of_text * one_tensor[0] if prev_start_of_text is not None else None
        prev_tokens = _pad_to_max_length(active_segments, generation_config.pad_token_id, padding='left', bos_token_tensor=prev_ids, cut_off_length=cut_off_length)
        decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
        kwargs['decoder_attention_mask'] = decoder_input_ids != generation_config.pad_token_id
    elif prompt_ids is not None:
        prev_tokens = prompt_ids[None].repeat(decoder_input_ids.shape[0], 1)
        decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
        kwargs.pop('decoder_attention_mask', None)
    else:
        kwargs.pop('decoder_attention_mask', None)
    return (decoder_input_ids, kwargs)