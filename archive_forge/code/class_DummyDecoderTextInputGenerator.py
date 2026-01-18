import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyDecoderTextInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy decoder text inputs.
    """
    SUPPORTED_INPUT_NAMES = ('decoder_input_ids', 'decoder_attention_mask')