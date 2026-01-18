import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
@add_start_docstrings('\n    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class BigBirdForSequenceClassification(BigBirdPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BigBirdModel(config)
        self.classifier = BigBirdClassificationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, BigBirdForSequenceClassification
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
        >>> model = BigBirdForSequenceClassification.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
        >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT

        >>> LONG_ARTICLE = squad_ds[81514]["context"]
        >>> inputs = tokenizer(LONG_ARTICLE, return_tensors="pt")
        >>> # long input article
        >>> list(inputs["input_ids"].shape)
        [1, 919]

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits
        >>> predicted_class_id = logits.argmax().item()
        >>> model.config.id2label[predicted_class_id]
        'LABEL_0'
        ```

        ```python
        >>> num_labels = len(model.config.id2label)
        >>> model = BigBirdForSequenceClassification.from_pretrained(
        ...     "l-yohai/bigbird-roberta-base-mnli", num_labels=num_labels
        ... )
        >>> labels = torch.tensor(1)
        >>> loss = model(**inputs, labels=labels).loss
        >>> round(loss.item(), 2)
        1.13
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)