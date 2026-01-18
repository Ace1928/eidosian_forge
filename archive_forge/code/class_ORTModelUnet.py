import importlib
import logging
import os
import shutil
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from diffusers import (
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_end_docstrings
import onnxruntime as ort
from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..pipelines.diffusers.pipeline_latent_consistency import LatentConsistencyPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_utils import VaeImageProcessor
from ..utils import (
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
class ORTModelUnet(_ORTDiffusionModelPart):

    def __init__(self, session: ort.InferenceSession, parent_model: ORTModel):
        super().__init__(session, parent_model)

    def forward(self, sample: np.ndarray, timestep: np.ndarray, encoder_hidden_states: np.ndarray, text_embeds: Optional[np.ndarray]=None, time_ids: Optional[np.ndarray]=None, timestep_cond: Optional[np.ndarray]=None):
        onnx_inputs = {'sample': sample, 'timestep': timestep, 'encoder_hidden_states': encoder_hidden_states}
        if text_embeds is not None:
            onnx_inputs['text_embeds'] = text_embeds
        if time_ids is not None:
            onnx_inputs['time_ids'] = time_ids
        if timestep_cond is not None:
            onnx_inputs['timestep_cond'] = timestep_cond
        outputs = self.session.run(None, onnx_inputs)
        return outputs