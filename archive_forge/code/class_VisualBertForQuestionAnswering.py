import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_visual_bert import VisualBertConfig
@add_start_docstrings('\n    VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled\n    output) for VQA.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForQuestionAnswering
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

        text = "Who is eating the apple?"
        inputs = tokenizer(text, return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        scores = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        index_to_gather = attention_mask.sum(1) - 2
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        index_to_gather = index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        pooled_output = torch.gather(sequence_output, 1, index_to_gather)
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)
        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction='batchmean')
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)