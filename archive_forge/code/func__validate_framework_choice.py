import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
@staticmethod
def _validate_framework_choice(framework: str):
    """
        Validates if the framework requested for the export is both correct and available, otherwise throws an
        exception.
        """
    if framework not in ['pt', 'tf']:
        raise ValueError(f'Only two frameworks are supported for ONNX export: pt or tf, but {framework} was provided.')
    elif framework == 'pt' and (not is_torch_available()):
        raise RuntimeError('Cannot export model to ONNX using PyTorch because no PyTorch package was found.')
    elif framework == 'tf' and (not is_tf_available()):
        raise RuntimeError('Cannot export model to ONNX using TensorFlow because no TensorFlow package was found.')