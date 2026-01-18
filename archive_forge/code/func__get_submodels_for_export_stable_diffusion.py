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
def _get_submodels_for_export_stable_diffusion(pipeline: 'StableDiffusionPipeline') -> Dict[str, Union['PreTrainedModel', 'ModelMixin']]:
    """
    Returns the components of a Stable Diffusion model.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline
    models_for_export = {}
    if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
        projection_dim = pipeline.text_encoder_2.config.projection_dim
    else:
        projection_dim = pipeline.text_encoder.config.projection_dim
    if pipeline.text_encoder is not None:
        if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
            pipeline.text_encoder.config.output_hidden_states = True
        models_for_export['text_encoder'] = pipeline.text_encoder
    is_torch_greater_or_equal_than_2_1 = version.parse(torch.__version__) >= version.parse('2.1.0')
    if not is_torch_greater_or_equal_than_2_1:
        pipeline.unet.set_attn_processor(AttnProcessor())
    pipeline.unet.config.text_encoder_projection_dim = projection_dim
    pipeline.unet.config.requires_aesthetics_score = getattr(pipeline.config, 'requires_aesthetics_score', False)
    models_for_export['unet'] = pipeline.unet
    vae_encoder = copy.deepcopy(pipeline.vae)
    if not is_torch_greater_or_equal_than_2_1:
        vae_encoder = override_diffusers_2_0_attn_processors(vae_encoder)
    vae_encoder.forward = lambda sample: {'latent_sample': vae_encoder.encode(x=sample)['latent_dist'].sample()}
    models_for_export['vae_encoder'] = vae_encoder
    vae_decoder = copy.deepcopy(pipeline.vae)
    if not is_torch_greater_or_equal_than_2_1:
        vae_decoder = override_diffusers_2_0_attn_processors(vae_decoder)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    models_for_export['vae_decoder'] = vae_decoder
    text_encoder_2 = getattr(pipeline, 'text_encoder_2', None)
    if text_encoder_2 is not None:
        text_encoder_2.config.output_hidden_states = True
        models_for_export['text_encoder_2'] = text_encoder_2
    return models_for_export