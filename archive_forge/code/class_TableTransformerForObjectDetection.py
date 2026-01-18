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
from .configuration_table_transformer import TableTransformerConfig
@add_start_docstrings('\n    Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on\n    top, for tasks such as COCO detection.\n    ', TABLE_TRANSFORMER_START_DOCSTRING)
class TableTransformerForObjectDetection(TableTransformerPreTrainedModel):

    def __init__(self, config: TableTransformerConfig):
        super().__init__(config)
        self.model = TableTransformerModel(config)
        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels + 1)
        self.bbox_predictor = TableTransformerMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3)
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(TABLE_TRANSFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TableTransformerObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[List[Dict]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], TableTransformerObjectDetectionOutput]:
        """
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        >>> import torch
        >>> from PIL import Image

        >>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        >>> image = Image.open(file_path).convert("RGB")

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        >>> model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected table with confidence 1.0 at location [202.1, 210.59, 1119.22, 385.09]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(pixel_values, pixel_mask=pixel_mask, decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        loss, loss_dict, auxiliary_outputs = (None, None, None)
        if labels is not None:
            matcher = TableTransformerHungarianMatcher(class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            losses = ['labels', 'boxes', 'cardinality']
            criterion = TableTransformerLoss(matcher=matcher, num_classes=self.config.num_labels, eos_coef=self.config.eos_coefficient, losses=losses)
            criterion.to(self.device)
            outputs_loss = {}
            outputs_loss['logits'] = logits
            outputs_loss['pred_boxes'] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
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
            return (loss, loss_dict) + output if loss is not None else output
        return TableTransformerObjectDetectionOutput(loss=loss, loss_dict=loss_dict, logits=logits, pred_boxes=pred_boxes, auxiliary_outputs=auxiliary_outputs, last_hidden_state=outputs.last_hidden_state, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)