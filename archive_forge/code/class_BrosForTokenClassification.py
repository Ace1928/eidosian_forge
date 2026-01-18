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
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_bros import BrosConfig
@add_start_docstrings('\n    Bros Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', BROS_START_DOCSTRING)
class BrosForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bros = BrosModel(config)
        classifier_dropout = config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, bbox: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, bbox_first_token_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """

        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bros(input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                loss = loss_fct(logits.view(-1, self.num_labels)[bbox_first_token_mask], labels.view(-1)[bbox_first_token_mask])
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)