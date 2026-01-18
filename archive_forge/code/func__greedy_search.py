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
def _greedy_search(self, input_ids: None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=None, logits_processor: Optional[FlaxLogitsProcessorList]=None, trace: bool=True, params: Optional[Dict[str, jnp.ndarray]]=None, model_kwargs: Optional[Dict[str, jnp.ndarray]]=None):
    max_length = max_length if max_length is not None else self.generation_config.max_length
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    batch_size, cur_len = input_ids.shape
    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
    cur_len = jnp.array(cur_len)
    sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
    is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    model = self.decode if self.config.is_encoder_decoder else self
    model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)
    state = GreedyState(cur_len=cur_len, sequences=sequences, running_token=input_ids, is_sent_finished=is_sent_finished, model_kwargs=model_kwargs)

    def greedy_search_cond_fn(state):
        """state termination condition fn."""
        has_reached_max_length = state.cur_len == max_length
        all_sequence_finished = jnp.all(state.is_sent_finished)
        finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
        return ~finish_generation

    def greedy_search_body_fn(state):
        """state update fn."""
        model_outputs = model(state.running_token, params=params, **state.model_kwargs)
        logits = model_outputs.logits[:, -1]
        logits = logits_processor(state.sequences, logits, state.cur_len)
        next_token = jnp.argmax(logits, axis=-1)
        next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]
        next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
        next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
        return GreedyState(cur_len=state.cur_len + 1, sequences=next_sequences, running_token=next_token, is_sent_finished=next_is_sent_finished, model_kwargs=next_model_kwargs)
    if input_ids.shape[1] > 1:
        state = greedy_search_body_fn(state)
    if not trace:
        state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
    else:
        state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
    return FlaxGreedySearchOutput(sequences=state.sequences)