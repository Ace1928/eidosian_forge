import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ....modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_open_llama import OpenLlamaConfig
@staticmethod
def _reorder_cache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
    return reordered_past