import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
def _prepare_4d_causal_attention_mask_for_sdpa_patched(attention_mask: Optional[torch.Tensor], input_shape: Union[torch.Size, Tuple, List], inputs_embeds: torch.Tensor, past_key_values_length: int, sliding_window: Optional[int]=None):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.

    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype)
    else:
        attention_mask = attn_mask_converter.to_causal_4d(input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    return attention_mask