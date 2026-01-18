import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig
from .logits_process import (
from .stopping_criteria import (
def _split_model_inputs(model_input: Union[ModelOutput, Dict], split_size: int, full_batch_size: int) -> List[Union[ModelOutput, Dict]]:
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    model_output_cls = type(model_input)
    if full_batch_size % split_size != 0:
        raise ValueError('`full_batch_size` must be divisible by `split_size`')
    if split_size > full_batch_size:
        raise ValueError('`split_size` must be smaller or equal to `full_batch_size`')
    keys = model_input.__dataclass_fields__.keys() if hasattr(model_input, '__dataclass_fields__') else model_input.keys()
    keys = [k for k in keys if k in model_input]
    bool_keys = [k for k in keys if isinstance(model_input[k], bool) or k == 'cache_position']
    keys_to_ignore = ['cache_position', 'encoder_outputs']
    non_bool_keys = [k for k in keys if not isinstance(model_input[k], bool) and k not in keys_to_ignore]
    data_split_list = [{k: _split(model_input[k], full_batch_size, split_size)[i] for k in non_bool_keys} for i in range(full_batch_size // split_size)]
    bool_data = {k: model_input[k] for k in bool_keys}
    if 'encoder_outputs' in model_input:
        encoder_outputs_split = _split_model_inputs(model_input['encoder_outputs'], split_size, full_batch_size)
        data_split_list = [{**data_split, 'encoder_outputs': encoder_outputs_split[i]} for i, data_split in enumerate(data_split_list)]
    split_model_inputs: List[Union[ModelOutput, Dict]] = [model_output_cls(**data_split, **bool_data) for data_split in data_split_list]
    return split_model_inputs