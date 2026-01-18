from typing import Optional
import torch
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.weight_utils import (default_weight_loader,

    Implementation for https://huggingface.co/Deci/DeciLM-7b-instruct.
    Based on the llama executor.

    The main difference is that DeciLM uses Variable Grouped Query Attention.
    The constant number of GQA heads in the decoder is overridden with a value
    per layer.

    Usually, in the HuggingFace implementation, instead of
    "config.num_key_value_heads", we use
    "config.num_key_value_heads_per_layer[i]" which varies.

    Currently, PagedAttention does not work well with variable GQA, so we
    normalize the weights upon loading, and use uniform GQA with the max value
    instead.
    