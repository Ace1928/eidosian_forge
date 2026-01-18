import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...utils import is_scipy_available
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_fnet import FNetConfig
@add_start_docstrings('FNet Model with a `next sentence prediction (classification)` head on top.', FNET_START_DOCSTRING)
class FNetForNextSentencePrediction(FNetPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.fnet = FNetModel(config)
        self.cls = FNetOnlyNSPHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, NextSentencePredictorOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FNetForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        >>> model = FNetForNextSentencePrediction.from_pretrained("google/fnet-base")
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```"""
        if 'next_sentence_label' in kwargs:
            warnings.warn('The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.', FutureWarning)
            labels = kwargs.pop('next_sentence_label')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.fnet(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)
        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (next_sentence_loss,) + output if next_sentence_loss is not None else output
        return NextSentencePredictorOutput(loss=next_sentence_loss, logits=seq_relationship_scores, hidden_states=outputs.hidden_states)