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
class SentenceTransformersCLIPPatcher(ModelPatcher):

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Dict[str, Any]):
        super().__init__(config, model, model_kwargs)

        def patched_forward(input_ids, attention_mask, pixel_values):
            vision_outputs = model[0].model.vision_model(pixel_values=pixel_values)
            image_embeds = model[0].model.visual_projection(vision_outputs[1])
            text_outputs = model[0].model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = model[0].model.text_projection(text_outputs[1])
            if len(model) > 1:
                image_embeds = model[1:](image_embeds)
                text_embeds = model[1:](text_embeds)
            return {'text_embeds': text_embeds, 'image_embeds': image_embeds}
        self.patched_forward = patched_forward