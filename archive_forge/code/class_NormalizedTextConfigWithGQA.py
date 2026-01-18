import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedTextConfigWithGQA(NormalizedTextConfig):
    NUM_KEY_VALUE_HEADS = 'num_key_value_heads'