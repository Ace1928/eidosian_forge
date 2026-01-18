import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
def _mask_pads(ll, smooth_obj):
    pad_mask = target.eq(self.config.generator.pad_token_id)
    if pad_mask.any():
        ll.masked_fill_(pad_mask, 0.0)
        smooth_obj.masked_fill_(pad_mask, 0.0)
    return (ll.squeeze(-1), smooth_obj.squeeze(-1))