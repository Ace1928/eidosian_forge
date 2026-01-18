import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
@add_start_docstrings('The bare OneFormer Model outputting raw hidden-states without any specific head on top.', ONEFORMER_START_DOCSTRING)
class OneFormerModel(OneFormerPreTrainedModel):
    main_input_name = ['pixel_values', 'task_inputs']

    def __init__(self, config: OneFormerConfig):
        super().__init__(config)
        self.pixel_level_module = OneFormerPixelLevelModule(config)
        self.transformer_module = OneFormerTransformerModule(in_features=config.conv_dim, config=config)
        self.task_encoder = OneFormerTaskModel(config)
        self.is_training = config.is_training
        if self.is_training:
            self.text_mapper = OneFormerTextMapper(config)
        else:
            self.text_mapper = None
        self.post_init()

    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, task_inputs: Tensor, text_inputs: Optional[Tensor]=None, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> OneFormerModelOutput:
        """
        Returns:
            `OneFormerModelOutput`
        Example:

        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import OneFormerProcessor, OneFormerModel

        >>> # download texting image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # load processor for preprocessing the inputs
        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> inputs = processor(image, ["semantic"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> mask_predictions = outputs.transformer_decoder_mask_predictions
        >>> class_predictions = outputs.transformer_decoder_class_predictions

        >>> f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}"
        '👉 Mask Predictions Shape: [1, 150, 128, 171], Class Predictions Shape: [1, 150, 151]'
        ```"""
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, _, height, width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states)
        multi_scale_features = pixel_level_module_output.decoder_features
        mask_features = pixel_level_module_output.decoder_last_feature
        task_token = self.task_encoder(task_inputs.to(self.dtype))
        if self.is_training:
            text_queries = self.text_mapper(text_inputs)
        else:
            text_queries = None
        transformer_module_output = self.transformer_module(multi_scale_features=multi_scale_features, mask_features=mask_features, task_token=task_token, output_attentions=output_attentions)
        queries = transformer_module_output.object_queries
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_features
            pixel_decoder_hidden_states = (pixel_level_module_output.decoder_last_feature,)
            for f in pixel_level_module_output.decoder_features:
                pixel_decoder_hidden_states += (f,)
            transformer_decoder_hidden_states = transformer_module_output.auxiliary_predictions
        output = OneFormerModelOutput(encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, transformer_decoder_object_queries=queries, transformer_decoder_contrastive_queries=transformer_module_output.contrastive_logits, transformer_decoder_mask_predictions=transformer_module_output.prediction_masks, transformer_decoder_class_predictions=transformer_module_output.prediction_class, transformer_decoder_auxiliary_predictions=transformer_module_output.auxiliary_predictions, text_queries=text_queries, task_token=task_token, attentions=transformer_module_output.attentions)
        if not return_dict:
            output = tuple((v for v in output.values()))
        return output