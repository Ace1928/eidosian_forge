import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_luke import LukeConfig
@add_start_docstrings('\n    The LUKE model with a span classification head on top (a linear layer on top of the hidden states output) for tasks\n    such as named entity recognition.\n    ', LUKE_START_DOCSTRING)
class LukeForEntitySpanClassification(LukePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.luke = LukeModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=EntitySpanClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, entity_ids: Optional[torch.LongTensor]=None, entity_attention_mask: Optional[torch.LongTensor]=None, entity_token_type_ids: Optional[torch.LongTensor]=None, entity_position_ids: Optional[torch.LongTensor]=None, entity_start_positions: Optional[torch.LongTensor]=None, entity_end_positions: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, EntitySpanClassificationOutput]:
        """
        entity_start_positions (`torch.LongTensor`):
            The start positions of entities in the word token sequence.

        entity_end_positions (`torch.LongTensor`):
            The end positions of entities in the word token sequence.

        labels (`torch.LongTensor` of shape `(batch_size, entity_length)` or `(batch_size, entity_length, num_labels)`, *optional*):
            Labels for computing the classification loss. If the shape is `(batch_size, entity_length)`, the cross
            entropy loss is used for the single-label classification. In this case, labels should contain the indices
            that should be in `[0, ..., config.num_labels - 1]`. If the shape is `(batch_size, entity_length,
            num_labels)`, the binary cross entropy loss is used for the multi-label classification. In this case,
            labels should only contain `[0, 1]`, where 0 and 1 indicate false and true, respectively.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LukeForEntitySpanClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
        >>> model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

        >>> text = "Beyoncé lives in Los Angeles"
        # List all possible entity spans in the text

        >>> word_start_positions = [0, 8, 14, 17, 21]  # character-based start positions of word tokens
        >>> word_end_positions = [7, 13, 16, 20, 28]  # character-based end positions of word tokens
        >>> entity_spans = []
        >>> for i, start_pos in enumerate(word_start_positions):
        ...     for end_pos in word_end_positions[i:]:
        ...         entity_spans.append((start_pos, end_pos))

        >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> predicted_class_indices = logits.argmax(-1).squeeze().tolist()
        >>> for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
        ...     if predicted_class_idx != 0:
        ...         print(text[span[0] : span[1]], model.config.id2label[predicted_class_idx])
        Beyoncé PER
        Los Angeles LOC
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_token_type_ids=entity_token_type_ids, entity_position_ids=entity_position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        hidden_size = outputs.last_hidden_state.size(-1)
        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        if entity_start_positions.device != outputs.last_hidden_state.device:
            entity_start_positions = entity_start_positions.to(outputs.last_hidden_state.device)
        start_states = torch.gather(outputs.last_hidden_state, -2, entity_start_positions)
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        if entity_end_positions.device != outputs.last_hidden_state.device:
            entity_end_positions = entity_end_positions.to(outputs.last_hidden_state.device)
        end_states = torch.gather(outputs.last_hidden_state, -2, entity_end_positions)
        feature_vector = torch.cat([start_states, end_states, outputs.entity_last_hidden_state], dim=2)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if labels.ndim == 2:
                loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))
        if not return_dict:
            return tuple((v for v in [loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions] if v is not None))
        return EntitySpanClassificationOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, entity_hidden_states=outputs.entity_hidden_states, attentions=outputs.attentions)