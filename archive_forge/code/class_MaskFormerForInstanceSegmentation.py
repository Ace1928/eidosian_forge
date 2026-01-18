import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerForInstanceSegmentation(MaskFormerPreTrainedModel):

    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.model = MaskFormerModel(config)
        hidden_size = config.decoder_config.hidden_size
        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = MaskformerMLPPredictionHead(hidden_size, hidden_size, config.mask_feature_size)
        self.matcher = MaskFormerHungarianMatcher(cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight)
        self.weight_dict: Dict[str, float] = {'loss_cross_entropy': config.cross_entropy_weight, 'loss_mask': config.mask_weight, 'loss_dice': config.dice_weight}
        self.criterion = MaskFormerLoss(config.num_labels, matcher=self.matcher, weight_dict=self.weight_dict, eos_coef=config.no_object_weight)
        self.post_init()

    def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, auxiliary_logits: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits)
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_logits(self, outputs: MaskFormerModelOutput) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        pixel_embeddings = outputs.pixel_decoder_last_hidden_state
        auxiliary_logits: List[str, Tensor] = []
        if self.config.use_auxiliary_loss:
            stacked_transformer_decoder_outputs = torch.stack(outputs.transformer_decoder_hidden_states)
            classes = self.class_predictor(stacked_transformer_decoder_outputs)
            class_queries_logits = classes[-1]
            mask_embeddings = self.mask_embedder(stacked_transformer_decoder_outputs)
            num_embeddings, batch_size, num_queries, num_channels = mask_embeddings.shape
            _, _, height, width = pixel_embeddings.shape
            binaries_masks = torch.zeros((num_embeddings, batch_size, num_queries, height, width), device=mask_embeddings.device)
            for c in range(num_channels):
                binaries_masks += mask_embeddings[..., c][..., None, None] * pixel_embeddings[None, :, None, c]
            masks_queries_logits = binaries_masks[-1]
            for aux_binary_masks, aux_classes in zip(binaries_masks[:-1], classes[:-1]):
                auxiliary_logits.append({'masks_queries_logits': aux_binary_masks, 'class_queries_logits': aux_classes})
        else:
            transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state
            classes = self.class_predictor(transformer_decoder_hidden_states)
            class_queries_logits = classes
            mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states)
            batch_size, num_queries, num_channels = mask_embeddings.shape
            _, _, height, width = pixel_embeddings.shape
            masks_queries_logits = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
            for c in range(num_channels):
                masks_queries_logits += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]
        return (class_queries_logits, masks_queries_logits, auxiliary_logits)

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskFormerForInstanceSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, mask_labels: Optional[List[Tensor]]=None, class_labels: Optional[List[Tensor]]=None, pixel_mask: Optional[Tensor]=None, output_auxiliary_logits: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> MaskFormerForInstanceSegmentationOutput:
        """
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:

        Examples:

        Semantic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to image_processor for postprocessing
        >>> predicted_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]

        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        >>> list(predicted_semantic_map.shape)
        [512, 683]
        ```

        Panoptic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on COCO panoptic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to image_processor for postprocessing
        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        >>> predicted_panoptic_map = result["segmentation"]
        >>> list(predicted_panoptic_map.shape)
        [480, 640]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        raw_outputs = self.model(pixel_values, pixel_mask, output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss, return_dict=return_dict, output_attentions=output_attentions)
        outputs = MaskFormerModelOutput(encoder_last_hidden_state=raw_outputs[0], pixel_decoder_last_hidden_state=raw_outputs[1], transformer_decoder_last_hidden_state=raw_outputs[2], encoder_hidden_states=raw_outputs[3] if output_hidden_states else None, pixel_decoder_hidden_states=raw_outputs[4] if output_hidden_states else None, transformer_decoder_hidden_states=raw_outputs[5] if output_hidden_states else None, hidden_states=raw_outputs[6] if output_hidden_states else None, attentions=raw_outputs[-1] if output_attentions else None)
        loss, loss_dict, auxiliary_logits = (None, None, None)
        class_queries_logits, masks_queries_logits, auxiliary_logits = self.get_logits(outputs)
        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits)
            loss = self.get_loss(loss_dict)
        output_auxiliary_logits = self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        if not output_auxiliary_logits:
            auxiliary_logits = None
        if not return_dict:
            output = tuple((v for v in (loss, class_queries_logits, masks_queries_logits, auxiliary_logits, *outputs.values()) if v is not None))
            return output
        return MaskFormerForInstanceSegmentationOutput(loss=loss, **outputs, class_queries_logits=class_queries_logits, masks_queries_logits=masks_queries_logits, auxiliary_logits=auxiliary_logits)