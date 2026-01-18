import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
@add_start_docstrings('Reformer Model with a `language modeling` head on top.', REFORMER_START_DOCSTRING)
class ReformerForMaskedLM(ReformerPreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder.weight', 'lm_head.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        assert not config.is_decoder, 'If you want to use `ReformerForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention.'
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, num_hashes: Optional[int]=None, labels: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels

        Returns:

        <Tip warning={true}>

        This example uses a false checkpoint since we don't have any available pretrained model for the masked language
        modeling task with the Reformer architecture.

        </Tip>

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, ReformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
        >>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")

        >>> # add mask_token
        >>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})  # doctest: +IGNORE_RESULT
        >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

        >>> # resize model's embedding matrix
        >>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)  # doctest: +IGNORE_RESULT

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> predicted_token = tokenizer.decode(predicted_token_id)
        ```

        ```python
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
        >>> # mask labels of non-[MASK] tokens
        >>> labels = torch.where(
        ...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
        ... )

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = round(outputs.loss.item(), 2)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        reformer_outputs = self.reformer(input_ids, position_ids=position_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, num_hashes=num_hashes, use_cache=False, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=logits, hidden_states=reformer_outputs.hidden_states, attentions=reformer_outputs.attentions)