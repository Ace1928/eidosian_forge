import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
    """Tie or clone module weights depending of whether we are using TorchScript or not"""
    if self.config.torchscript:
        output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
    else:
        output_embeddings.weight = input_embeddings.weight
    if getattr(output_embeddings, 'bias', None) is not None:
        output_embeddings.bias.data = nn.functional.pad(output_embeddings.bias.data, (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]), 'constant', 0)
    if hasattr(output_embeddings, 'out_features') and hasattr(input_embeddings, 'num_embeddings'):
        output_embeddings.out_features = input_embeddings.num_embeddings