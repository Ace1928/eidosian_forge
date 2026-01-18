import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@staticmethod
def _infer_library_from_model(model: Union['PreTrainedModel', 'TFPreTrainedModel'], library_name: Optional[str]=None):
    if library_name is not None:
        return library_name
    if hasattr(model, 'pretrained_cfg') or hasattr(model.config, 'pretrained_cfg') or hasattr(model.config, 'architecture'):
        library_name = 'timm'
    elif hasattr(model.config, '_diffusers_version') or getattr(model, 'config_name', '') == 'model_index.json':
        library_name = 'diffusers'
    elif hasattr(model, '_model_config'):
        library_name = 'sentence_transformers'
    else:
        library_name = 'transformers'
    return library_name