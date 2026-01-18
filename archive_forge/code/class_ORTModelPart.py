from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from onnxruntime import InferenceSession
from ..utils import NormalizedConfigManager
from ..utils.logging import warn_once
from .utils import get_ordered_input_names, logging
class ORTModelPart:
    """
    For multi-file ONNX models, such as encoder-decoder models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """

    def __init__(self, session: InferenceSession, parent_model: 'ORTModel'):
        self.session = session
        self.parent_model = parent_model
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(self.parent_model.config.model_type)(self.parent_model.config)
        self.main_input_name = self.parent_model.main_input_name
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self._ordered_input_names = get_ordered_input_names(self.input_names.keys(), func=self.forward)

    @property
    def device(self):
        return self.parent_model.device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)