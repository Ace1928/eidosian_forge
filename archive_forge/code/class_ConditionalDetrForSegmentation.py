import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_conditional_detr import ConditionalDetrConfig
@add_start_docstrings('\n    CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top,\n    for tasks such as COCO panoptic.\n\n    ', CONDITIONAL_DETR_START_DOCSTRING)
class ConditionalDetrForSegmentation(ConditionalDetrPreTrainedModel):

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.conditional_detr = ConditionalDetrForObjectDetection(config)
        hidden_size, number_of_heads = (config.d_model, config.encoder_attention_heads)
        intermediate_channel_sizes = self.conditional_detr.model.backbone.conv_encoder.intermediate_channel_sizes
        self.mask_head = ConditionalDetrMaskHeadSmallConv(hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size)
        self.bbox_attention = ConditionalDetrMHAttentionMap(hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std)
        self.post_init()

    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[List[dict]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], ConditionalDetrSegmentationOutput]:
        """
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
            dictionary containing at least the following 3 keys: 'class_labels', 'boxes' and 'masks' (the class labels,
            bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
            should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, the boxes a
            `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)` and the masks a
            `torch.FloatTensor` of shape `(number of bounding boxes in the image, height, width)`.

        Returns:

        Examples:

        ```python
        >>> import io
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy

        >>> from transformers import (
        ...     AutoImageProcessor,
        ...     ConditionalDetrConfig,
        ...     ConditionalDetrForSegmentation,
        ... )
        >>> from transformers.image_transforms import rgb_to_id

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")

        >>> # randomly initialize all weights of the model
        >>> config = ConditionalDetrConfig()
        >>> model = ConditionalDetrForSegmentation(config)

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # Use the `post_process_panoptic_segmentation` method of the `image_processor` to retrieve post-processed panoptic segmentation maps
        >>> # Segmentation results are returned as a list of dictionaries
        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])
        >>> # A tensor of shape (height, width) where each value denotes a segment id, filled with -1 if no segment is found
        >>> panoptic_seg = result[0]["segmentation"]
        >>> # Get prediction score and segment_id to class_id mapping of each segment
        >>> panoptic_segments_info = result[0]["segments_info"]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)
        features, object_queries_list = self.conditional_detr.model.backbone(pixel_values, pixel_mask=pixel_mask)
        feature_map, mask = features[-1]
        batch_size, num_channels, height, width = feature_map.shape
        projected_feature_map = self.conditional_detr.model.input_projection(feature_map)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)
        if encoder_outputs is None:
            encoder_outputs = self.conditional_detr.model.encoder(inputs_embeds=flattened_features, attention_mask=flattened_mask, object_queries=object_queries, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        query_position_embeddings = self.conditional_detr.model.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)
        decoder_outputs = self.conditional_detr.model.decoder(inputs_embeds=queries, attention_mask=None, object_queries=object_queries, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=flattened_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        logits = self.conditional_detr.class_labels_classifier(sequence_output)
        pred_boxes = self.conditional_detr.bbox_predictor(sequence_output).sigmoid()
        memory = encoder_outputs[0].permute(0, 2, 1).view(batch_size, self.config.d_model, height, width)
        mask = flattened_mask.view(batch_size, height, width)
        bbox_mask = self.bbox_attention(sequence_output, memory, mask=~mask)
        seg_masks = self.mask_head(projected_feature_map, bbox_mask, [features[2][0], features[1][0], features[0][0]])
        pred_masks = seg_masks.view(batch_size, self.conditional_detr.config.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        loss, loss_dict, auxiliary_outputs = (None, None, None)
        if labels is not None:
            matcher = ConditionalDetrHungarianMatcher(class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            losses = ['labels', 'boxes', 'cardinality', 'masks']
            criterion = ConditionalDetrLoss(matcher=matcher, num_classes=self.config.num_labels, focal_alpha=self.config.focal_alpha, losses=losses)
            criterion.to(self.device)
            outputs_loss = {}
            outputs_loss['logits'] = logits
            outputs_loss['pred_boxes'] = pred_boxes
            outputs_loss['pred_masks'] = pred_masks
            if self.config.auxiliary_loss:
                intermediate = decoder_outputs.intermediate_hidden_states if return_dict else decoder_outputs[-1]
                outputs_class = self.conditional_detr.class_labels_classifier(intermediate)
                outputs_coord = self.conditional_detr.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self.conditional_detr._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {'loss_ce': 1, 'loss_bbox': self.config.bbox_loss_coefficient}
            weight_dict['loss_giou'] = self.config.giou_loss_coefficient
            weight_dict['loss_mask'] = self.config.mask_loss_coefficient
            weight_dict['loss_dice'] = self.config.dice_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum((loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes, pred_masks) + auxiliary_outputs + decoder_outputs + encoder_outputs
            else:
                output = (logits, pred_boxes, pred_masks) + decoder_outputs + encoder_outputs
            return (loss, loss_dict) + output if loss is not None else output
        return ConditionalDetrSegmentationOutput(loss=loss, loss_dict=loss_dict, logits=logits, pred_boxes=pred_boxes, pred_masks=pred_masks, auxiliary_outputs=auxiliary_outputs, last_hidden_state=decoder_outputs.last_hidden_state, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)