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
class DecoderModelPatcher(ModelPatcher):

    def __enter__(self):
        super().__enter__()
        if AttentionMaskConverter is not None:
            AttentionMaskConverter._make_causal_mask = _make_causal_mask_patched_staticmethod
            if _transformers_version >= version.parse('4.36'):
                AttentionMaskConverter._unmask_unattended = _unmask_unattended_patched_staticmethod
        if _transformers_version >= version.parse('4.36'):
            patch_everywhere('_prepare_4d_causal_attention_mask_for_sdpa', _prepare_4d_causal_attention_mask_for_sdpa_patched)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if AttentionMaskConverter is not None:
            AttentionMaskConverter._make_causal_mask = staticmethod(self.original_make_causal)
            if _transformers_version >= version.parse('4.36'):
                AttentionMaskConverter._unmask_unattended = staticmethod(self.original_unmask_unattended)
        if _transformers_version >= version.parse('4.36'):
            patch_everywhere('_prepare_4d_causal_attention_mask_for_sdpa', self.original_prepare_4d_causal_attention_mask_for_sdpa)

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None):
        super().__init__(config, model, model_kwargs)
        if _transformers_version >= version.parse('4.36'):
            self.original_prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa
            self.original_unmask_unattended = AttentionMaskConverter._unmask_unattended
        if AttentionMaskConverter is not None:
            self.original_make_causal = AttentionMaskConverter._make_causal_mask