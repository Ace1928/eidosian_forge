import copy
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ..models.auto import (
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
def _get_logits_processor(self, generation_config: GenerationConfig, input_ids_seq_length: int, logits_processor: Optional[FlaxLogitsProcessorList]) -> FlaxLogitsProcessorList:
    """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
    processors = FlaxLogitsProcessorList()
    if generation_config.min_length is not None and generation_config.eos_token_id is not None and (generation_config.min_length > -1):
        processors.append(FlaxMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
    if generation_config.forced_bos_token_id is not None:
        processors.append(FlaxForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
    if generation_config.forced_eos_token_id is not None:
        processors.append(FlaxForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id))
    if generation_config.suppress_tokens is not None:
        processors.append(FlaxSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
    if generation_config.begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = begin_index if input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None else begin_index + 1
        if generation_config.forced_decoder_ids is not None and len(generation_config.forced_decoder_ids) > 0:
            begin_index += generation_config.forced_decoder_ids[-1][0]
        processors.append(FlaxSuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index))
    if generation_config.forced_decoder_ids is not None:
        forced_decoder_ids = [[input_ids_seq_length + i[0] - 1, i[1]] for i in generation_config.forced_decoder_ids]
        processors.append(FlaxForceTokensLogitsProcessor(forced_decoder_ids))
    processors = self._merge_criteria_processor_list(processors, logits_processor)
    return processors