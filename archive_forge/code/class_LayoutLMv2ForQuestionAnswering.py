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
@add_start_docstrings('\n    LayoutLMv2 Model with a span classification head on top for extractive question-answering tasks such as\n    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to\n    compute `span start logits` and `span end logits`).\n    ', LAYOUTLMV2_START_DOCSTRING)
class LayoutLMv2ForQuestionAnswering(LayoutLMv2PreTrainedModel):

    def __init__(self, config, has_visual_segment_embedding=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.has_visual_segment_embedding = has_visual_segment_embedding
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, bbox: Optional[torch.LongTensor]=None, image: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        In this example below, we give the LayoutLMv2 model an image (of texts) and ask it a question. It will give us
        a prediction of what it thinks the answer is (the span of the answer within the texts parsed from the image).

        ```python
        >>> from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering, set_seed
        >>> import torch
        >>> from PIL import Image
        >>> from datasets import load_dataset

        >>> set_seed(88)
        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

        >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
        >>> image_path = dataset["test"][0]["file"]
        >>> image = Image.open(image_path).convert("RGB")
        >>> question = "When is coffee break?"
        >>> encoding = processor(image, question, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> predicted_start_idx = outputs.start_logits.argmax(-1).item()
        >>> predicted_end_idx = outputs.end_logits.argmax(-1).item()
        >>> predicted_start_idx, predicted_end_idx
        (154, 287)

        >>> predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
        >>> predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
        >>> predicted_answer  # results are not very good without further fine-tuning
        'council mem - bers conducted by trrf treasurer philip g. kuehn to get answers which the public ...
        ```

        ```python
        >>> target_start_index = torch.tensor([7])
        >>> target_end_index = torch.tensor([14])
        >>> outputs = model(**encoding, start_positions=target_start_index, end_positions=target_end_index)
        >>> predicted_answer_span_start = outputs.start_logits.argmax(-1).item()
        >>> predicted_answer_span_end = outputs.end_logits.argmax(-1).item()
        >>> predicted_answer_span_start, predicted_answer_span_end
        (154, 287)
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
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)