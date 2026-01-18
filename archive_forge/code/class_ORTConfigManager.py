import os
import re
from enum import Enum
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from packaging import version
from transformers.utils import logging
import onnxruntime as ort
from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss
from ..utils.import_utils import _is_package_available
class ORTConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.

    Attributes:
        _conf (`Dict[str]`):
            A dictionary mapping each supported model type to the corresponding ONNX Runtime model type.
    """
    _conf = {'albert': 'bert', 'bart': 'bart', 'bert': 'bert', 'big-bird': 'bert', 'blenderbot': 'bert', 'bloom': 'gpt2', 'camembert': 'bert', 'codegen': 'gpt2', 'deberta': 'bert', 'deberta-v2': 'bert', 'distilbert': 'bert', 'electra': 'bert', 'gpt2': 'gpt2', 'gpt-bigcode': 'gpt2', 'gpt-neo': 'gpt2', 'gpt-neox': 'gpt2', 'gptj': 'gpt2', 'longt5': 'bert', 'llama': 'gpt2', 'marian': 'bart', 'mbart': 'bart', 'mistral': 'gpt2', 'mt5': 'bart', 'm2m-100': 'bart', 'nystromformer': 'bert', 'pegasus': 'bert', 'roberta': 'bert', 't5': 'bert', 'vit': 'vit', 'whisper': 'bart', 'xlm-roberta': 'bert'}

    @classmethod
    def get_model_ort_type(cls, model_type: str) -> str:
        model_type = model_type.replace('_', '-')
        cls.check_supported_model(model_type)
        return cls._conf[model_type]

    @classmethod
    def check_supported_model(cls, model_type: str):
        if model_type not in cls._conf:
            model_types = ', '.join(cls._conf.keys())
            raise KeyError(f'{model_type} model type is not supported yet. Only {model_types} are supported. If you want to support {model_type} please propose a PR or open up an issue.')

    @classmethod
    def check_optimization_supported_model(cls, model_type: str, optimization_config):
        supported_model_types_for_optimization = ['bart', 'bert', 'gpt2', 'tnlr', 't5', 'unet', 'vae', 'clip', 'vit', 'swin']
        model_type = model_type.replace('_', '-')
        if model_type not in cls._conf or cls._conf[model_type] not in supported_model_types_for_optimization:
            raise NotImplementedError(f"ONNX Runtime doesn't support the graph optimization of {model_type} yet. Only {list(cls._conf.keys())} are supported. If you want to support {model_type} please propose a PR or open up an issue in ONNX Runtime: https://github.com/microsoft/onnxruntime.")