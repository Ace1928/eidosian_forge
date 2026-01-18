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
@add_start_docstrings('The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood loss.', REALM_START_DOCSTRING)
class RealmKnowledgeAugEncoder(RealmPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder']

    def __init__(self, config):
        super().__init__(config)
        self.realm = RealmBertModel(self.config)
        self.cls = RealmOnlyMLMHead(self.config)
        self.post_init()

    def get_input_embeddings(self):
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.realm.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format('batch_size, num_candidates, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, relevance_score: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, mlm_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, MaskedLMOutput]:
        """
        relevance_score (`torch.FloatTensor` of shape `(batch_size, num_candidates)`, *optional*):
            Relevance score derived from RealmScorer, must be specified if you want to compute the masked language
            modeling loss.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        mlm_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid calculating joint loss on certain positions. If not specified, the loss will not be masked.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, RealmKnowledgeAugEncoder

        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")
        >>> model = RealmKnowledgeAugEncoder.from_pretrained(
        ...     "google/realm-cc-news-pretrained-encoder", num_candidates=2
        ... )

        >>> # batch_size = 2, num_candidates = 2
        >>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

        >>> inputs = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        flattened_input_ids, flattened_attention_mask, flattened_token_type_ids = self._flatten_inputs(input_ids, attention_mask, token_type_ids)
        joint_outputs = self.realm(flattened_input_ids, attention_mask=flattened_attention_mask, token_type_ids=flattened_token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        joint_output = joint_outputs[0]
        prediction_scores = self.cls(joint_output)
        candidate_score = relevance_score
        masked_lm_loss = None
        if labels is not None:
            if candidate_score is None:
                raise ValueError('You have to specify `relevance_score` when `labels` is specified in order to compute loss.')
            batch_size, seq_length = labels.size()
            if mlm_mask is None:
                mlm_mask = torch.ones_like(labels, dtype=torch.float32)
            else:
                mlm_mask = mlm_mask.type(torch.float32)
            loss_fct = CrossEntropyLoss(reduction='none')
            mlm_logits = prediction_scores.view(-1, self.config.vocab_size)
            mlm_targets = labels.tile(1, self.config.num_candidates).view(-1)
            masked_lm_log_prob = -loss_fct(mlm_logits, mlm_targets).view(batch_size, self.config.num_candidates, seq_length)
            candidate_log_prob = candidate_score.log_softmax(-1).unsqueeze(-1)
            joint_gold_log_prob = candidate_log_prob + masked_lm_log_prob
            marginal_gold_log_probs = joint_gold_log_prob.logsumexp(1)
            masked_lm_loss = -torch.nansum(torch.sum(marginal_gold_log_probs * mlm_mask) / torch.sum(mlm_mask))
        if not return_dict:
            output = (prediction_scores,) + joint_outputs[2:4]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=joint_outputs.hidden_states, attentions=joint_outputs.attentions)