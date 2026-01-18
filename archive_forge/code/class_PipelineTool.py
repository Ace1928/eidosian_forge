import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
from .agent_types import handle_agent_inputs, handle_agent_outputs
from {module_name} import {class_name}
class PipelineTool(Tool):
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    need to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs (additional keyword arguments, *optional*):
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """
    pre_processor_class = AutoProcessor
    model_class = None
    post_processor_class = AutoProcessor
    default_checkpoint = None

    def __init__(self, model=None, pre_processor=None, post_processor=None, device=None, device_map=None, model_kwargs=None, token=None, **hub_kwargs):
        if not is_torch_available():
            raise ImportError('Please install torch in order to use this tool.')
        if not is_accelerate_available():
            raise ImportError('Please install accelerate in order to use this tool.')
        if model is None:
            if self.default_checkpoint is None:
                raise ValueError('This tool does not implement a default checkpoint, you need to pass one.')
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model
        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs['device_map'] = device_map
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs['token'] = token
        super().__init__()

    def setup(self):
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        if isinstance(self.pre_processor, str):
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)
        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)
        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)
        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = PartialState().default_device
        if self.device_map is None:
            self.model.to(self.device)
        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        return self.post_processor(outputs)

    def __call__(self, *args, **kwargs):
        args, kwargs = handle_agent_inputs(*args, **kwargs)
        if not self.is_initialized:
            self.setup()
        encoded_inputs = self.encode(*args, **kwargs)
        encoded_inputs = send_to_device(encoded_inputs, self.device)
        outputs = self.forward(encoded_inputs)
        outputs = send_to_device(outputs, 'cpu')
        decoded_outputs = self.decode(outputs)
        return handle_agent_outputs(decoded_outputs, self.outputs)