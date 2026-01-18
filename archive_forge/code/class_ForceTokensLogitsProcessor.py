import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class ForceTokensLogitsProcessor(LogitsProcessor):
    """
    This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before generation. The processor will set their log probs to `inf` so that they are
    sampled at their corresponding index. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:
    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # This Whisper model forces the generation to start with `50362` at the first position by default, i.e.
    >>> # `"forced_decoder_ids": [[1, 50362]]`. This means all other tokens are masked out.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(
    ...     all(outputs.scores[0][0, i] == float("-inf") for i in range(processor.tokenizer.vocab_size) if i != 50362)
    ... )
    True
    >>> print(outputs.scores[0][0, 50362])
    tensor(0.)

    >>> # If we disable `forced_decoder_ids`, we stop seeing that effect
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, forced_decoder_ids=None)
    >>> print(
    ...     all(outputs.scores[0][0, i] == float("-inf") for i in range(processor.tokenizer.vocab_size) if i != 50362)
    ... )
    False
    >>> print(outputs.scores[0][0, 50362])
    tensor(19.3140)
    ```
    """

    def __init__(self, force_token_map: List[List[int]]):
        self.force_token_map = dict(force_token_map)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generation_idx = input_ids.shape[-1]
        current_token = self.force_token_map.get(generation_idx, None)
        if current_token is not None:
            scores[:, :] = -float('inf')
            scores[:, current_token] = 0
        return scores