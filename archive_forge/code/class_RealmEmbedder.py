import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
@add_start_docstrings('The embedder of REALM outputting projected score that will be used to calculate relevance score.', REALM_START_DOCSTRING)
class RealmEmbedder(RealmPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        self.realm = RealmBertModel(self.config)
        self.cls = RealmScorerProjection(self.config)
        self.post_init()

    def get_input_embeddings(self):
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.realm.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=RealmEmbedderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RealmEmbedderOutput]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RealmEmbedder
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
        >>> model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> projected_score = outputs.projected_score
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        realm_outputs = self.realm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooler_output = realm_outputs[1]
        projected_score = self.cls(pooler_output)
        if not return_dict:
            return (projected_score,) + realm_outputs[2:4]
        else:
            return RealmEmbedderOutput(projected_score=projected_score, hidden_states=realm_outputs.hidden_states, attentions=realm_outputs.attentions)