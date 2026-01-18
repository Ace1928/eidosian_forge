import copy
import logging
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
import onnxruntime as ort
from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..utils import check_if_transformers_greater
from ..utils.file_utils import validate_file_exists
from ..utils.normalized_config import NormalizedConfigManager
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .base import ORTDecoderForSeq2Seq, ORTEncoder
from .constants import (
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
from huggingface_hub.utils import EntryNotFoundError
@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForPix2Struct(ORTModelForConditionalGeneration, GenerationMixin):
    """
    Pix2struct model with a language modeling head for ONNX Runtime inference. This class officially supports pix2struct.
    """
    auto_model_class = Pix2StructForConditionalGeneration
    main_input_name = 'flattened_patches'

    def _initialize_encoder(self, session: ort.InferenceSession) -> ORTEncoder:
        return ORTEncoderForPix2Struct(session, self)

    @add_start_docstrings_to_model_forward(PIX2STRUCT_ONNX_MODEL_DOCSTRING + PIX2STRUCT_EXAMPLE.format(processor_class=_PROCESSOR_FOR_DOC, model_class='ORTModelForPix2Struct', checkpoint='google/pix2struct-ai2d-base'))
    def forward(self, flattened_patches: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, labels: Optional[torch.LongTensor]=None, **kwargs) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(flattened_patches=flattened_patches, attention_mask=attention_mask)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(torch.int64)
        else:
            attention_mask = attention_mask.astype(np.int64)
        model = self.decoder if past_key_values is None or not self.use_cache or self.use_merged else self.decoder_with_past
        decoder_outputs = model(input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, past_key_values=past_key_values, encoder_hidden_states=encoder_outputs.last_hidden_state, encoder_attention_mask=attention_mask, labels=labels)
        return Seq2SeqLMOutput(loss=decoder_outputs.get('loss', None), logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values)

    def prepare_inputs_for_generation(self, input_ids, flattened_patches: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs) -> Dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)
        return {'flattened_patches': flattened_patches, 'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def get_encoder(self) -> ORTEncoder:
        return self.encoder

    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        ORTModelForSeq2SeqLM._reorder_cache(past, beam_idx)

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True