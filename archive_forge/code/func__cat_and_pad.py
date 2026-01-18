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
@staticmethod
def _cat_and_pad(tensors, pad_token_id):
    output = tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
    ind = 0
    for t in tensors:
        output[ind:ind + t.shape[0], :t.shape[1]] = t
        ind += t.shape[0]
    return output