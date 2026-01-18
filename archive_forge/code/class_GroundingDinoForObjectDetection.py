import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
@add_start_docstrings('\n    Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top,\n    for tasks such as COCO detection.\n    ', GROUNDING_DINO_START_DOCSTRING)
class GroundingDinoForObjectDetection(GroundingDinoPreTrainedModel):
    _tied_weights_keys = ['bbox_embed\\.[1-9]\\d*', 'model\\.decoder\\.bbox_embed\\.[0-9]\\d*']

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)
        self.model = GroundingDinoModel(config)
        _class_embed = GroundingDinoContrastiveEmbedding(config)
        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3)
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            for _ in range(config.decoder_layers):
                _bbox_embed = GroundingDinoMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3)
                self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroundingDinoObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, input_ids: torch.LongTensor, token_type_ids: torch.LongTensor=None, attention_mask: torch.LongTensor=None, pixel_mask: Optional[torch.BoolTensor]=None, encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]]=None):
        """
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, GroundingDinoForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        >>> model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to COCO API
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = processor.image_processor.post_process_object_detection(
        ...     outputs, threshold=0.35, target_sizes=target_sizes
        ... )[0]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 1) for i in box.tolist()]
        ...     print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
        Detected 1 with confidence 0.45 at location [344.8, 23.2, 637.4, 373.8]
        Detected 1 with confidence 0.41 at location [11.9, 51.6, 316.6, 472.9]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, pixel_mask=pixel_mask, encoder_outputs=encoder_outputs, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        idx = 5 + (1 if output_attentions else 0) + (1 if output_hidden_states else 0)
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[idx]
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference_points = outputs.init_reference_points if return_dict else outputs[1]
        inter_references_points = outputs.intermediate_reference_points if return_dict else outputs[3]
        outputs_classes = []
        outputs_coords = []
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-05)
            outputs_class = self.class_embed[level](vision_hidden_state=hidden_states[:, level], text_hidden_state=enc_text_hidden_state, text_token_mask=attention_mask.bool())
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f'reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}')
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        loss, loss_dict, auxiliary_outputs = (None, None, None)
        if labels is not None:
            matcher = GroundingDinoHungarianMatcher(class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            losses = ['labels', 'boxes', 'cardinality']
            criterion = GroundingDinoLoss(matcher=matcher, num_classes=self.config.num_labels, focal_alpha=self.config.focal_alpha, losses=losses)
            criterion.to(self.device)
            outputs_loss = {}
            outputs_loss['logits'] = logits
            outputs_loss['pred_boxes'] = pred_boxes
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
            if self.config.two_stage:
                enc_outputs_coord = outputs[-1].sigmoid()
                outputs_loss['enc_outputs'] = {'logits': outputs[-2], 'pred_boxes': enc_outputs_coord}
            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {'loss_ce': 1, 'loss_bbox': self.config.bbox_loss_coefficient}
            weight_dict['loss_giou'] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum((loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = (loss, loss_dict) + output if loss is not None else output
            return tuple_outputs
        dict_outputs = GroundingDinoObjectDetectionOutput(loss=loss, loss_dict=loss_dict, logits=logits, pred_boxes=pred_boxes, last_hidden_state=outputs.last_hidden_state, auxiliary_outputs=auxiliary_outputs, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, encoder_last_hidden_state_vision=outputs.encoder_last_hidden_state_vision, encoder_last_hidden_state_text=outputs.encoder_last_hidden_state_text, encoder_vision_hidden_states=outputs.encoder_vision_hidden_states, encoder_text_hidden_states=outputs.encoder_text_hidden_states, encoder_attentions=outputs.encoder_attentions, intermediate_hidden_states=outputs.intermediate_hidden_states, intermediate_reference_points=outputs.intermediate_reference_points, init_reference_points=outputs.init_reference_points, enc_outputs_class=outputs.enc_outputs_class, enc_outputs_coord_logits=outputs.enc_outputs_coord_logits)
        return dict_outputs