import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
def _get_new_num_tokens_layer(self, new_num_tokens, layer):
    embeddings = self.get_input_embeddings()
    if layer == -1:
        layer = len(embeddings.emb_layers) - 1
    assert 0 <= layer <= len(embeddings.emb_layers) - 1
    new_num_tokens_layer = new_num_tokens - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[:layer]]) - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[layer + 1:]])
    return (new_num_tokens_layer, layer)