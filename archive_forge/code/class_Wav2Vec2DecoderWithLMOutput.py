import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends
@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2DecoderWithLM`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        logit_score (list of `float` or `float`):
            Total logit score of the beams associated with produced text.
        lm_score (list of `float`):
            Fused lm_score of the beams associated with produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """
    text: Union[List[List[str]], List[str], str]
    logit_score: Union[List[List[float]], List[float], float] = None
    lm_score: Union[List[List[float]], List[float], float] = None
    word_offsets: Union[List[List[ListOfDict]], List[ListOfDict], ListOfDict] = None