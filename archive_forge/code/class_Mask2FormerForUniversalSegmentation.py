import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
@add_start_docstrings('The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.', MASK2FORMER_START_DOCSTRING)
class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = 'pixel_values'

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)
        self.weight_dict: Dict[str, float] = {'loss_cross_entropy': config.class_weight, 'loss_mask': config.mask_weight, 'loss_dice': config.dice_weight}
        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()

    def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, auxiliary_predictions: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, mask_labels=mask_labels, class_labels=class_labels, auxiliary_predictions=auxiliary_predictions)
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []
        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({'masks_queries_logits': aux_binary_masks, 'class_queries_logits': aux_classes})
        return auxiliary_logits

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, mask_labels: Optional[List[Tensor]]=None, class_labels: Optional[List[Tensor]]=None, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_auxiliary_logits: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Mask2FormerForUniversalSegmentationOutput:
        """
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `Mask2FormerUniversalSegmentationOutput`

        Examples:

        Instance segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-coco-instance"
        ... )

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get instance segmentation map
        >>> pred_instance_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> print(pred_instance_map.shape)
        torch.Size([480, 640])
        ```

        Semantic segmentation example:
        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on ADE20k semantic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get semantic segmentation map
        >>> pred_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> print(pred_semantic_map.shape)
        torch.Size([512, 683])
        ```

        Panoptic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on CityScapes panoptic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-cityscapes-panoptic"
        ... )

        >>> url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get panoptic segmentation map
        >>> pred_panoptic_map = image_processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> print(pred_panoptic_map.shape)
        torch.Size([338, 676])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss, output_attentions=output_attentions, return_dict=True)
        loss, loss_dict, auxiliary_logits = (None, None, None)
        class_queries_logits = ()
        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)
        masks_queries_logits = outputs.masks_queries_logits
        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(masks_queries_logits=masks_queries_logits[-1], class_queries_logits=class_queries_logits[-1], mask_labels=mask_labels, class_labels=class_labels, auxiliary_predictions=auxiliary_logits)
            loss = self.get_loss(loss_dict)
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states
        output_auxiliary_logits = self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        if not output_auxiliary_logits:
            auxiliary_logits = None
        output = Mask2FormerForUniversalSegmentationOutput(loss=loss, class_queries_logits=class_queries_logits[-1], masks_queries_logits=masks_queries_logits[-1], auxiliary_logits=auxiliary_logits, encoder_last_hidden_state=outputs.encoder_last_hidden_state, pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state, transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state, encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, attentions=outputs.attentions)
        if not return_dict:
            output = tuple((v for v in output.values() if v is not None))
            if loss is not None:
                output = loss + output
        return output