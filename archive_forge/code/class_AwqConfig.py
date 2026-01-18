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
class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 )
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
        do_fuse (`bool`, *optional*, defaults to `False`):
            Whether to fuse attention and mlp layers together for faster inference
        fuse_max_seq_len (`int`, *optional*):
            The Maximum sequence length to generate when using fusing.
        modules_to_fuse (`dict`, *optional*, default to `None`):
            Overwrite the natively supported fusing scheme with the one specified by the users.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
    """

    def __init__(self, bits: int=4, group_size: int=128, zero_point: bool=True, version: AWQLinearVersion=AWQLinearVersion.GEMM, backend: AwqBackendPackingMethod=AwqBackendPackingMethod.AUTOAWQ, do_fuse: Optional[bool]=None, fuse_max_seq_len: Optional[int]=None, modules_to_fuse: Optional[dict]=None, modules_to_not_convert: Optional[List]=None, **kwargs):
        self.quant_method = QuantizationMethod.AWQ
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.backend = backend
        self.fuse_max_seq_len = fuse_max_seq_len
        self.modules_to_not_convert = modules_to_not_convert
        self.modules_to_fuse = modules_to_fuse
        if do_fuse is None:
            self.do_fuse = modules_to_fuse is not None and len(modules_to_fuse) > 0
        else:
            self.do_fuse = do_fuse
        self.fuse_max_seq_len = fuse_max_seq_len
        self.post_init()

    def post_init(self):
        """
        Safety checker that arguments are correct
        """
        if not torch.cuda.is_available():
            raise ValueError('AWQ is only available on GPU')
        if self.backend not in [AwqBackendPackingMethod.AUTOAWQ, AwqBackendPackingMethod.LLMAWQ]:
            raise ValueError(f'Only supported quantization backends in {AwqBackendPackingMethod.AUTOAWQ} and {AwqBackendPackingMethod.LLMAWQ} - not recognized backend {self.backend}')
        self.version = AWQLinearVersion.from_str(self.version)
        if self.version not in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV]:
            raise ValueError(f'Only supported versions are in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV] - not recognized version {self.version}')
        if self.backend == AwqBackendPackingMethod.LLMAWQ:
            compute_capability = torch.cuda.get_device_capability()
            major, minor = compute_capability
            if major < 8:
                raise ValueError('LLM-AWQ backend is only supported on GPUs with compute capability >= 8.0')
        if self.do_fuse and self.fuse_max_seq_len is None:
            raise ValueError('You cannot enable fused modules without specifying a `fuse_max_seq_len`, make sure to pass a valid `fuse_max_seq_len` for your usecase')
        if self.do_fuse:
            awq_version_supports_fusing = False
            MIN_AWQ_VERSION = '0.1.7'
            if is_auto_awq_available():
                awq_version_supports_fusing = version.parse(importlib.metadata.version('autoawq')) >= version.parse(MIN_AWQ_VERSION)
            if not awq_version_supports_fusing:
                raise ValueError(f'You current version of `autoawq` does not support module fusing, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}.')
        if self.modules_to_not_convert is not None:
            awq_version_supports_non_conversion = False
            MIN_AWQ_VERSION = '0.1.8'
            if is_auto_awq_available():
                awq_version_supports_non_conversion = version.parse(importlib.metadata.version('autoawq')) >= version.parse(MIN_AWQ_VERSION)
            if not awq_version_supports_non_conversion:
                raise ValueError(f'You current version of `autoawq` does not support module quantization skipping, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}.')
        if self.do_fuse and self.modules_to_fuse is not None:
            required_keys = ['hidden_size', 'num_attention_heads', 'num_key_value_heads', 'mlp', 'attention', 'layernorm', 'use_alibi']
            if not all((key in self.modules_to_fuse for key in required_keys)):
                raise ValueError(f'Required fields are missing in the fusing mapping, required fields are {required_keys}')

    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ['do_fuse', 'modules_to_fuse', 'fuse_max_seq_len']
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict