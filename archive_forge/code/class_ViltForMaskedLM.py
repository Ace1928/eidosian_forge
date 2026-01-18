import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vilt import ViltConfig
@add_start_docstrings('\n    ViLT Model with a language modeling head on top as done during pretraining.\n    ', VILT_START_DOCSTRING)
class ViltForMaskedLM(ViltPreTrainedModel):
    _tied_weights_keys = ['mlm_score.decoder.weight', 'mlm_score.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        self.vilt = ViltModel(config)
        self.mlm_score = ViltMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        """
        labels (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in *[-100, 0, ...,
            config.vocab_size]* (see *input_ids* docstring) Tokens with indices set to *-100* are ignored (masked), the
            loss is only computed for the tokens with labels in *[0, ..., config.vocab_size]*

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForMaskedLM
        >>> import requests
        >>> from PIL import Image
        >>> import re
        >>> import torch

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a bunch of [MASK] laying on a [MASK]."

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)

        >>> tl = len(re.findall("\\[MASK\\]", text))
        >>> inferred_token = [text]

        >>> # gradually fill in the MASK tokens, one by one
        >>> with torch.no_grad():
        ...     for i in range(tl):
        ...         encoded = processor.tokenizer(inferred_token)
        ...         input_ids = torch.tensor(encoded.input_ids)
        ...         encoded = encoded["input_ids"][0][1:-1]
        ...         outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
        ...         mlm_logits = outputs.logits[0]  # shape (seq_len, vocab_size)
        ...         # only take into account text features (minus CLS and SEP token)
        ...         mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
        ...         mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
        ...         # only take into account text
        ...         mlm_values[torch.tensor(encoded) != 103] = 0
        ...         select = mlm_values.argmax().item()
        ...         encoded[select] = mlm_ids[select].item()
        ...         inferred_token = [processor.decode(encoded)]

        >>> selected_token = ""
        >>> encoded = processor.tokenizer(inferred_token)
        >>> output = processor.decode(encoded.input_ids[0], skip_special_tokens=True)
        >>> print(output)
        a bunch of cats laying on a couch.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vilt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, image_embeds=image_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output, pooled_output = outputs[:2]
        text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        text_features, _ = (sequence_output[:, :text_seq_len], sequence_output[:, text_seq_len:])
        mlm_logits = self.mlm_score(text_features)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(mlm_logits.device)
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (mlm_logits,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=mlm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)