import re
from typing import List, Optional, Tuple, Union
import numpy as np
from ..utils import (
from .base import ChunkPipeline, build_pipeline_init_args
from .question_answering import select_starts_ends
def _sanitize_parameters(self, padding=None, doc_stride=None, max_question_len=None, lang: Optional[str]=None, tesseract_config: Optional[str]=None, max_answer_len=None, max_seq_len=None, top_k=None, handle_impossible_answer=None, timeout=None, **kwargs):
    preprocess_params, postprocess_params = ({}, {})
    if padding is not None:
        preprocess_params['padding'] = padding
    if doc_stride is not None:
        preprocess_params['doc_stride'] = doc_stride
    if max_question_len is not None:
        preprocess_params['max_question_len'] = max_question_len
    if max_seq_len is not None:
        preprocess_params['max_seq_len'] = max_seq_len
    if lang is not None:
        preprocess_params['lang'] = lang
    if tesseract_config is not None:
        preprocess_params['tesseract_config'] = tesseract_config
    if timeout is not None:
        preprocess_params['timeout'] = timeout
    if top_k is not None:
        if top_k < 1:
            raise ValueError(f'top_k parameter should be >= 1 (got {top_k})')
        postprocess_params['top_k'] = top_k
    if max_answer_len is not None:
        if max_answer_len < 1:
            raise ValueError(f'max_answer_len parameter should be >= 1 (got {max_answer_len}')
        postprocess_params['max_answer_len'] = max_answer_len
    if handle_impossible_answer is not None:
        postprocess_params['handle_impossible_answer'] = handle_impossible_answer
    return (preprocess_params, {}, postprocess_params)