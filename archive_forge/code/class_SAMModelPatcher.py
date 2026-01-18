import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
class SAMModelPatcher(ModelPatcher):

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None):
        super().__init__(config, model, model_kwargs)

        def patched_forward(pixel_values=None, input_points=None, input_labels=None, image_embeddings=None, image_positional_embeddings=None, return_dict=True, **kwargs):
            if config.variant == 'monolith':
                return self.orig_forward(pixel_values=pixel_values, input_points=input_points, input_labels=input_labels, image_embeddings=image_embeddings, return_dict=return_dict, **kwargs)
            elif config.variant == 'split':
                if config.vision_encoder:
                    image_positional_embeddings = model.get_image_wide_positional_embeddings()
                    batch_size = pixel_values.shape[0]
                    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
                    vision_outputs = model.vision_encoder(pixel_values, output_attentions=False, output_hidden_states=False, return_dict=return_dict)
                    image_embeddings = vision_outputs[0]
                    if not return_dict:
                        return (image_embeddings, image_positional_embeddings)
                    else:
                        return {'image_embeddings': image_embeddings, 'image_positional_embeddings': image_positional_embeddings}
                else:
                    if input_points is None:
                        raise ValueError('input_points is required to export the prompt encoder / mask decoder.')
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(input_points=input_points, input_labels=input_labels, input_boxes=None, input_masks=None)
                    low_res_masks, iou_predictions, _ = model.mask_decoder(image_embeddings=image_embeddings, image_positional_embeddings=image_positional_embeddings, sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=True, attention_similarity=None, target_embedding=None, output_attentions=False)
                    if not return_dict:
                        return (iou_predictions, low_res_masks)
                    else:
                        return {'iou_scores': iou_predictions, 'pred_masks': low_res_masks}
        self.patched_forward = patched_forward