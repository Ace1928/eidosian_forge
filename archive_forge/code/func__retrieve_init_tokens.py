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
def _retrieve_init_tokens(self, input_features, generation_config, config, num_segment_frames, kwargs):

    def replace_or_add(lst: List[int], num: int, itr: Iterator[int]):
        """short function to replace num with a itr in lst"""
        found = any((i in lst for i in itr))
        if found:
            lst = [num if i in itr else i for i in lst]
        else:
            lst.append(num)
        return lst
    task = getattr(generation_config, 'task', None)
    language = getattr(generation_config, 'language', None)
    if kwargs.get('forced_decoder_ids', None) is not None:
        forced_decoder_ids = kwargs['forced_decoder_ids']
    elif hasattr(generation_config, 'forced_decoder_ids') and generation_config.forced_decoder_ids is not None:
        forced_decoder_ids = generation_config.forced_decoder_ids
        if language is None and task is None and (forced_decoder_ids[0][1] is None):
            logger.warning_once("Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English.This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`.")
    elif hasattr(config, 'forced_decoder_ids') and config.forced_decoder_ids is not None:
        forced_decoder_ids = config.forced_decoder_ids
    else:
        forced_decoder_ids = None
    if forced_decoder_ids is not None and task is not None:
        logger.info(f'You have passed task={task}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of task={task}.')
        forced_decoder_ids = None
    elif forced_decoder_ids is not None and language is not None:
        logger.info(f'You have passed language={language}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of language={language}.')
        forced_decoder_ids = None
    init_tokens = [generation_config.decoder_start_token_id]
    if forced_decoder_ids is not None and forced_decoder_ids[0][0] == 1:
        i = 1
        while len(forced_decoder_ids) > 0 and forced_decoder_ids[0][0] == i:
            init_tokens += [forced_decoder_ids[0][1]]
            forced_decoder_ids = forced_decoder_ids[1:]
            i += 1
        if len(forced_decoder_ids) > 0:
            warnings.warn(f'You are using token ids in `forced_decoder_ids` that do not seem to correctly follow the prompt pattern of Whisper. Make sure that {forced_decoder_ids} has an entry for all indices >= 1 and < {forced_decoder_ids[0][0]}. `forced_decoder_ids` will be passed as a logit processor, but note that this functionality has been deprecated and will throw an error in v4.39.', FutureWarning)
        generation_config.forced_decoder_ids = forced_decoder_ids if len(forced_decoder_ids) > 0 else None
    is_lang_id_undefined = len(init_tokens) <= 1 or (len(init_tokens) > 1 and init_tokens[1] is None)
    if language is not None:
        if language in generation_config.lang_to_id.keys():
            language_token = language
        elif language in TO_LANGUAGE_CODE.keys():
            language_token = f'<|{TO_LANGUAGE_CODE[language]}|>'
        elif language in TO_LANGUAGE_CODE.values():
            language_token = f'<|{language}|>'
        else:
            is_language_code = len(language) == 2
            raise ValueError(f'Unsupported language: {language}. Language should be one of: {(list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys()))}.')
        if language_token not in generation_config.lang_to_id:
            raise ValueError(f'{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`.(You should just add it to the generation config)')
        lang_id = generation_config.lang_to_id[language_token]
        replace_or_add(init_tokens, lang_id, generation_config.lang_to_id.values())
    elif hasattr(generation_config, 'lang_to_id') and is_lang_id_undefined:
        lang_ids = self.detect_language(input_features=input_features, encoder_outputs=kwargs.get('encoder_outputs', None), generation_config=generation_config, num_segment_frames=num_segment_frames)
        if torch.unique(lang_ids).shape[0] > 1:
            raise ValueError("Multiple languages detected when trying to predict the most likely target language for transcription. It is currently not supported to transcribe to different languages in a single batch. Please make sure to either force a single language by passing `language='...'` or make sure all input audio is of the same language.")
        lang_id = lang_ids[0].item()
        if len(init_tokens) > 1:
            init_tokens[1] = lang_id
        else:
            init_tokens.append(lang_id)
    if task is not None:
        if task in TASK_IDS:
            init_tokens.append(generation_config.task_to_id[generation_config.task])
            task_id = generation_config.task_to_id[generation_config.task]
            replace_or_add(init_tokens, task_id, generation_config.task_to_id.values())
        else:
            raise ValueError(f'The `{task}`task is not supported. The task should be one of `{TASK_IDS}`')
    elif language is not None and hasattr(generation_config, 'task_to_id'):
        if not any((i in init_tokens for i in generation_config.task_to_id.values())):
            init_tokens.append(generation_config.task_to_id['transcribe'])
    if not generation_config.return_timestamps and hasattr(generation_config, 'no_timestamps_token_id') and (init_tokens[-1] != generation_config.no_timestamps_token_id):
        init_tokens.append(generation_config.no_timestamps_token_id)
    elif generation_config.return_timestamps and init_tokens[-1] == generation_config.no_timestamps_token_id:
        logger.info("<|notimestamps|> prompt token is removed from generation_config since `return_timestamps` is set to `'True'`.")
        init_tokens = init_tokens[:-1]
    init_tokens = [t for t in init_tokens if t is not None]
    return init_tokens