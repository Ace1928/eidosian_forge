import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
@add_start_docstrings('\n    LayoutLMv2 Model with a token classification head on top (a linear layer on top of the text part of the hidden\n    states) e.g. for sequence labeling (information extraction) tasks such as\n    [FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://rrc.cvc.uab.es/?ch=13),\n    [CORD](https://github.com/clovaai/cord) and [Kleister-NDA](https://github.com/applicaai/kleister-nda).\n    ', LAYOUTLMV2_START_DOCSTRING)
class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, bbox: Optional[torch.LongTensor]=None, image: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, LayoutLMv2ForTokenClassification, set_seed
        >>> from PIL import Image
        >>> from datasets import load_dataset

        >>> set_seed(88)

        >>> datasets = load_dataset("nielsr/funsd", split="test")
        >>> labels = datasets.features["ner_tags"].feature.names
        >>> id2label = {v: k for v, k in enumerate(labels)}

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        >>> model = LayoutLMv2ForTokenClassification.from_pretrained(
        ...     "microsoft/layoutlmv2-base-uncased", num_labels=len(labels)
        ... )

        >>> data = datasets[0]
        >>> image = Image.open(data["image_path"]).convert("RGB")
        >>> words = data["words"]
        >>> boxes = data["bboxes"]  # make sure to normalize your bounding boxes
        >>> word_labels = data["ner_tags"]
        >>> encoding = processor(
        ...     image,
        ...     words,
        ...     boxes=boxes,
        ...     word_labels=word_labels,
        ...     padding="max_length",
        ...     truncation=True,
        ...     return_tensors="pt",
        ... )

        >>> outputs = model(**encoding)
        >>> logits, loss = outputs.logits, outputs.loss

        >>> predicted_token_class_ids = logits.argmax(-1)
        >>> predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]
        >>> predicted_tokens_classes[:5]
        ['B-ANSWER', 'B-HEADER', 'B-HEADER', 'B-HEADER', 'B-HEADER']
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.layoutlmv2(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)