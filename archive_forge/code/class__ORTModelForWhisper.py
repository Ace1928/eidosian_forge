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
class _ORTModelForWhisper(ORTModelForSpeechSeq2Seq):
    """
    Whisper implements its own generate() method.
    """

    @classmethod
    def _from_pretrained(cls, model_id: Union[str, Path], config: 'PretrainedConfig', **kwargs):
        return super(ORTModelForSpeechSeq2Seq, cls)._from_pretrained(model_id, config, **kwargs)

    def generate(self, input_features: Optional[torch.Tensor]=None, generation_config=None, logits_processor=None, stopping_criteria=None, prefix_allowed_tokens_fn=None, synced_gpus=False, return_timestamps=None, task=None, language=None, is_multilingual=None, prompt_ids: Optional[torch.Tensor]=None, num_segment_frames: Optional[int]=None, return_token_timestamps: Optional[bool]=None, return_segments: bool=False, attention_mask: Optional[torch.Tensor]=None, time_precision: int=0.02, return_dict_in_generate: Optional[bool]=None, **kwargs):
        if 'inputs' in kwargs:
            input_features = kwargs.pop('inputs')
            warnings.warn('The input name `inputs` is deprecated. Please make sure to use `input_features` instead.', FutureWarning)
        return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        input_stride = 1 * 2
        if num_segment_frames is None:
            num_segment_frames = input_stride * self.config.max_source_positions
        if input_features is not None:
            total_input_frames = input_features.shape[-1]
        elif 'encoder_outputs' in kwargs:
            encoder_outputs_shape = kwargs['encoder_outputs'][0].shape if isinstance(kwargs['encoder_outputs'], BaseModelOutput) else kwargs['encoder_outputs'].shape
            total_input_frames = encoder_outputs_shape[1] * input_stride
        else:
            raise ValueError('Make sure to provide either `input_features` or `encoder_outputs` to `generate`.')
        is_shortform = total_input_frames <= num_segment_frames
        if return_timestamps is True:
            if not hasattr(generation_config, 'no_timestamps_token_id'):
                raise ValueError('You are trying to return timestamps, but the generation config is not properly set. Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363')
            generation_config.return_timestamps = return_timestamps
        elif not is_shortform:
            if return_timestamps is False:
                raise ValueError('You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features.')
            if not hasattr(generation_config, 'no_timestamps_token_id'):
                raise ValueError('You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which requires the generation config to have `no_timestamps_token_id` correctly. Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363or make sure to pass no more than 3000 mel input features.')
            logger.info('Setting `return_timestamps=True` for long-form generation.')
            generation_config.return_timestamps = True
        else:
            generation_config.return_timestamps = False
        if is_multilingual is not None:
            if not hasattr(generation_config, 'is_multilingual'):
                raise ValueError('The generation config is outdated and is thus not compatible with the `is_multilingual` argument to `generate`. Please update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224')
            generation_config.is_multilingual = is_multilingual
        if hasattr(generation_config, 'is_multilingual') and (not generation_config.is_multilingual):
            if task is not None or language is not None:
                raise ValueError('Cannot specify `task` or `language` for an English-only model. If the model is intended to be multilingual, pass `is_multilingual=True` to generate, or update the generation config.')
        if language is not None:
            if not hasattr(generation_config, 'lang_to_id'):
                raise ValueError('The generation config is outdated and is thus not compatible with the `language` argument to `generate`. Either set the language using the `forced_decoder_ids` in the model config, or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224')
            language = language.lower()
            generation_config.language = language
        if task is not None:
            if not hasattr(generation_config, 'task_to_id'):
                raise ValueError('The generation config is outdated and is thus not compatible with the `task` argument to `generate`. Either set the task using the `forced_decoder_ids` in the model config, or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224')
            generation_config.task = task
        forced_decoder_ids = None
        if hasattr(self.config, 'forced_decoder_ids') and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif hasattr(self.generation_config, 'forced_decoder_ids') and self.generation_config.forced_decoder_ids is not None:
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get('forced_decoder_ids', None)
        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, 'language'):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f'<|{TO_LANGUAGE_CODE[generation_config.language]}|>'
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f'<|{generation_config.language}|>'
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(f'Unsupported language: {generation_config.language}. Language should be one of: {(list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys()))}.')
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))
            if hasattr(generation_config, 'task'):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(f'The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`')
            elif hasattr(generation_config, 'task_to_id'):
                forced_decoder_ids.append((2, generation_config.task_to_id['transcribe']))
            if hasattr(generation_config, 'no_timestamps_token_id') and (not generation_config.return_timestamps):
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))
        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids
        if prompt_ids is not None:
            if kwargs.get('decoder_start_token_id') is not None:
                raise ValueError('When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten.')
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1:]
            kwargs.update({'decoder_start_token_id': decoder_start_token_id})
            if kwargs.get('max_new_tokens', None) is not None:
                kwargs['max_new_tokens'] += len(text_prompt_ids)
                if kwargs['max_new_tokens'] >= self.config.max_target_positions:
                    raise ValueError(f'The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` {kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced `prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the `max_target_positions` of the Whisper model: {self.config.max_target_positions}. You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, so that their combined length is less that {self.config.max_target_positions}.')
            non_prompt_forced_decoder_ids = kwargs.pop('forced_decoder_ids', None) or generation_config.forced_decoder_ids
            forced_decoder_ids = [*text_prompt_ids, generation_config.decoder_start_token_id, *[token for _rank, token in non_prompt_forced_decoder_ids]]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids
        if return_token_timestamps:
            kwargs['output_attentions'] = True
            return_dict_in_generate = True
            if getattr(generation_config, 'task', None) == 'translate':
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, 'alignment_heads'):
                raise ValueError('Model generation config has no `alignment_heads`, token-level timestamps not available. See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config.')
            if kwargs.get('num_frames') is not None:
                generation_config.num_frames = kwargs.pop('num_frames')
        if generation_config.return_timestamps is True:
            last_forced_decoder_ids = generation_config.forced_decoder_ids[-1][-1] if hasattr(self.config, 'forced_decoder_ids') and self.config.forced_decoder_ids else None
            if last_forced_decoder_ids == self.generation_config.no_timestamps_token_id:
                forced_decoder_ids = generation_config.forced_decoder_ids[:-1]
                generation_config.forced_decoder_ids = None if len(forced_decoder_ids) == 0 else forced_decoder_ids
            timestamp_processor = [WhisperTimeStampLogitsProcessor(generation_config)]
            logits_processor = timestamp_processor if logits_processor is None else timestamp_processor + logits_processor
        if is_shortform:
            outputs = super().generate(input_features, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, return_dict_in_generate=return_dict_in_generate, **kwargs)
            if return_token_timestamps and hasattr(generation_config, 'alignment_heads'):
                num_frames = getattr(generation_config, 'num_frames', None)
                outputs['token_timestamps'] = self._extract_token_timestamps(outputs, generation_config.alignment_heads, num_frames=num_frames)
            return outputs
        if not return_segments and return_dict_in_generate:
            raise ValueError("Make sure to set `return_segments=True` to return generation outputs as part of the `'segments' key.`")
        timestamp_begin = self.generation_config.no_timestamps_token_id + 1
        batch_size = input_features.shape[0]
        if batch_size > 1 and attention_mask is None:
            raise ValueError('When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` ')
        elif batch_size > 1:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((1,), dtype=torch.long)
        current_segments = [[] for _ in range(batch_size)]
        cur_to_prev_index_map = list(range(batch_size))
        cur_bsz = prev_bsz = batch_size
        while (seek < max_frames).any():
            prev_bsz = cur_bsz
            new_cur_to_prev_index_map = []
            for i in range(prev_bsz):
                prev_i = cur_to_prev_index_map[i]
                if seek[prev_i] >= max_frames[prev_i]:
                    cut_index = i + (cur_bsz - prev_bsz)
                    cur_bsz -= 1
                    input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1:]], dim=0)
                else:
                    new_cur_to_prev_index_map.append(prev_i)
            cur_to_prev_index_map = new_cur_to_prev_index_map
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)
            segment_input = []
            for i in range(cur_bsz):
                prev_i = cur_to_prev_index_map[i]
                segment_input_slice = input_features[i:i + 1, :, seek[prev_i]:seek[prev_i] + seek_num_frames[prev_i]]
                if segment_input_slice.shape[-1] < num_segment_frames:
                    segment_input_slice = torch.nn.functional.pad(segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1]))
                segment_input.append(segment_input_slice)
            segment_input = torch.cat(segment_input, dim=0)
            seek_outputs = super().generate(segment_input, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, return_dict_in_generate=return_dict_in_generate, **kwargs)
            if return_token_timestamps and hasattr(generation_config, 'alignment_heads'):
                num_frames = getattr(generation_config, 'num_frames', None)
                seek_outputs['token_timestamps'] = self._extract_token_timestamps(seek_outputs, generation_config.alignment_heads, num_frames=num_frames)
            if return_dict_in_generate:
                seek_sequences = seek_outputs['sequences']
                seek_outputs = [{k: v[i] for k, v in seek_outputs.items()} for i in range(next(iter(seek_outputs.values())).size(0))]
            else:
                seek_sequences = seek_outputs
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = cur_to_prev_index_map[i]
                is_not_final = seek[prev_i] + num_segment_frames < max_frames[prev_i]
                if is_not_final and seek_sequence[-1] == self.generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]
                if seek_sequence[-1] == self.generation_config.pad_token_id:
                    num_paddings = (seek_sequence == self.generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]
                segments, segment_offset = self._retrieve_segment(seek_sequence=seek_sequence, seek_outputs=seek_outputs, time_offset=time_offset, timestamp_begin=timestamp_begin, seek_num_frames=seek_num_frames, cur_bsz=cur_bsz, time_precision=time_precision, input_stride=input_stride, prev_idx=prev_i, idx=i)
                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset
        sequences = []
        max_total_length = 0
        for current_segment_list in current_segments:
            sequences.append(torch.cat([d['tokens'] for d in current_segment_list], dim=-1))
            max_total_length = max(max_total_length, len(sequences[-1]))
        for i in range(batch_size):
            sequences[i] = torch.nn.functional.pad(sequences[i], pad=(0, max_total_length - len(sequences[i])), value=self.generation_config.pad_token_id)
        sequences = torch.stack(sequences, dim=0)
        if return_segments:
            return {'sequences': sequences, 'segments': current_segments}
        return sequences