import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
import onnxruntime
from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..onnx.utils import check_model_uses_external_data
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ..utils.save_utils import maybe_save_preprocessors
from .constants import DECODER_MERGED_ONNX_FILE_PATTERN, DECODER_ONNX_FILE_PATTERN, DECODER_WITH_PAST_ONNX_FILE_PATTERN
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .models.bloom import bloom_convert_to_bloom_cache, bloom_convert_to_standard_cache
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_WEIGHTS_NAME
class ORTGPTBigCodeForCausalLM(ORTModelForCausalLM):

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
        model_inputs = {'input_ids': input_ids}
        model_inputs.update({'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask})
        return model_inputs