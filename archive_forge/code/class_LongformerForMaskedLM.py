import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longformer import LongformerConfig
@add_start_docstrings('Longformer Model with a `language modeling` head on top.', LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder']

    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerMaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Mask filling example:

        ```python
        >>> from transformers import AutoTokenizer, LongformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        ```

        Let's try a very long input.

        ```python
        >>> TXT = (
        ...     "My friends are <mask> but they eat too many carbs."
        ...     + " That's why I decide not to eat with them." * 300
        ... )
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['healthy', 'skinny', 'thin', 'good', 'vegetarian']
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return LongformerMaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)