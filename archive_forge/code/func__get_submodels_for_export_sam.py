import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
from transformers.utils import is_tf_available, is_torch_available
from ...utils import (
from ...utils.import_utils import _diffusers_version
from ..tasks import TasksManager
from .constants import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME
def _get_submodels_for_export_sam(model, variant):
    models_for_export = {}
    if variant == 'monolith':
        models_for_export['model'] = model
    else:
        models_for_export['vision_encoder'] = model
        models_for_export['prompt_encoder_mask_decoder'] = model
    return models_for_export