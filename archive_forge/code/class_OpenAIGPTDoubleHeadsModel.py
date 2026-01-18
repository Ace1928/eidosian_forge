import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu_new, silu
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
from .configuration_openai import OpenAIGPTConfig
@add_start_docstrings('\nOpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for\nRocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the\ninput embeddings, the classification head takes as input the input of a specified classification token index in the\ninput sequence).\n', OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, mc_token_ids: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, mc_labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], OpenAIGPTDoubleHeadsModelOutput]:
        """
        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
            1]`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-1, 0, ..., config.vocab_size]` All labels set to `-100` are
            ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

        Return:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, OpenAIGPTDoubleHeadsModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
        >>> model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-community/openai-gpt")
        >>> tokenizer.add_special_tokens(
        ...     {"cls_token": "[CLS]"}
        ... )  # Add a [CLS] to the vocabulary (we should train it also!)
        >>> model.resize_token_embeddings(len(tokenizer))

        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        >>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        >>> mc_token_ids = torch.tensor([input_ids.size(-1) - 1, input_ids.size(-1) - 1]).unsqueeze(0)  # Batch size 1

        >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
        >>> lm_logits = outputs.logits
        >>> mc_logits = outputs.mc_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        lm_loss, mc_loss = (None, None)
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return (lm_loss,) + output if lm_loss is not None else output
        return OpenAIGPTDoubleHeadsModelOutput(loss=lm_loss, mc_loss=mc_loss, logits=lm_logits, mc_logits=mc_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)