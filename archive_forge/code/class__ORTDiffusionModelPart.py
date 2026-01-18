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
class _ORTDiffusionModelPart:
    """
    For multi-file ONNX models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """
    CONFIG_NAME = 'config.json'

    def __init__(self, session: ort.InferenceSession, parent_model: ORTModel):
        self.session = session
        self.parent_model = parent_model
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        config_path = Path(session._model_path).parent / self.CONFIG_NAME
        self.config = self.parent_model._dict_from_json_file(config_path) if config_path.is_file() else {}
        self.input_dtype = {inputs.name: _ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    @property
    def device(self):
        return self.parent_model.device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)