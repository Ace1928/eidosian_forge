import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
@dataclass
class QuantoConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `quanto`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
        activations (`str`, *optional*):
            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__(self, weights='int8', activations=None, modules_to_not_convert: Optional[List]=None, **kwargs):
        self.quant_method = QuantizationMethod.QUANTO
        self.weights = weights
        self.activations = activations
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        """
        Safety checker that arguments are correct
        """
        accepted_weights = ['float8', 'int8', 'int4', 'int2']
        accepted_activations = [None, 'int8', 'float8']
        if self.weights not in accepted_weights:
            raise ValueError(f'Only support weights in {accepted_weights} but found {self.weights}')
        if self.activations not in accepted_activations:
            raise ValueError(f'Only support weights in {accepted_activations} but found {self.activations}')