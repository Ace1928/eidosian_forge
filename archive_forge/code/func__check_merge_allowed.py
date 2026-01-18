from __future__ import annotations
import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
def _check_merge_allowed(self):
    """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
    if getattr(self.model, 'quantization_method', None) == 'gptq':
        raise ValueError('Cannot merge LORA layers when the model is gptq quantized')
    if self.peft_config.get('layer_replication'):
        raise ValueError('Cannot merge LORA layers when base model layers are replicated')