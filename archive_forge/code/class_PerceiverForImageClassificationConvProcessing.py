import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
@add_start_docstrings('\nExample use of Perceiver for image classification, for tasks such as ImageNet.\n\nThis model uses a 2D conv+maxpool preprocessing network. As shown in the paper, this model can achieve a top-1 accuracy\nof 82.1 on ImageNet.\n\n[`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]\n(with `prep_type="conv"`) to preprocess the input images, and\n[`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of\n[`PerceiverModel`] into classification logits.\n', PERCEIVER_START_DOCSTRING)
class PerceiverForImageClassificationConvProcessing(PerceiverPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        fourier_position_encoding_kwargs_preprocessor = {'concat_pos': True, 'max_resolution': (56, 56), 'num_bands': 64, 'sine_only': False}
        trainable_position_encoding_kwargs_decoder = {'num_channels': config.d_latents, 'index_dims': 1}
        self.num_labels = config.num_labels
        self.perceiver = PerceiverModel(config, input_preprocessor=PerceiverImagePreprocessor(config, prep_type='conv', spatial_downsample=1, position_encoding_type='fourier', fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor), decoder=PerceiverClassificationDecoder(config, num_channels=config.d_latents, trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder, use_query_residual=True))
        self.post_init()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, inputs: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None, pixel_values: Optional[torch.Tensor]=None) -> Union[Tuple, PerceiverClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, PerceiverForImageClassificationConvProcessing
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-conv")
        >>> model = PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")

        >>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
        >>> outputs = model(inputs=inputs)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 1000]

        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: tabby, tabby cat
        ```"""
        if inputs is not None and pixel_values is not None:
            raise ValueError('You cannot use both `inputs` and `pixel_values`')
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.perceiver(inputs=inputs, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = outputs.logits if return_dict else outputs[0]
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
        return PerceiverClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)