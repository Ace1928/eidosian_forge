import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedTextConfig(NormalizedConfig):
    VOCAB_SIZE = 'vocab_size'
    HIDDEN_SIZE = 'hidden_size'
    NUM_LAYERS = 'num_hidden_layers'
    NUM_ATTENTION_HEADS = 'num_attention_heads'
    EOS_TOKEN_ID = 'eos_token_id'