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
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input time. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(self, load_in_8bit=False, load_in_4bit=False, llm_int8_threshold=6.0, llm_int8_skip_modules=None, llm_int8_enable_fp32_cpu_offload=False, llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype=None, bnb_4bit_quant_type='fp4', bnb_4bit_use_double_quant=False, **kwargs):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES
        if load_in_4bit and load_in_8bit:
            raise ValueError('load_in_4bit and load_in_8bit are both True, but only one can be used at the same time')
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError('bnb_4bit_compute_dtype must be a string or a torch.dtype')
        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if self.load_in_8bit and value:
            raise ValueError('load_in_4bit and load_in_8bit are both True, but only one can be used at the same time')
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if self.load_in_4bit and value:
            raise ValueError('load_in_4bit and load_in_8bit are both True, but only one can be used at the same time')
        self._load_in_8bit = value

    def post_init(self):
        """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.llm_int8_threshold, float):
            raise ValueError('llm_int8_threshold must be a float')
        if self.llm_int8_skip_modules is not None and (not isinstance(self.llm_int8_skip_modules, list)):
            raise ValueError('llm_int8_skip_modules must be a list of strings')
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise ValueError('llm_int8_enable_fp32_cpu_offload must be a boolean')
        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise ValueError('llm_int8_has_fp16_weight must be a boolean')
        if self.bnb_4bit_compute_dtype is not None and (not isinstance(self.bnb_4bit_compute_dtype, torch.dtype)):
            raise ValueError('bnb_4bit_compute_dtype must be torch.dtype')
        if not isinstance(self.bnb_4bit_quant_type, str):
            raise ValueError('bnb_4bit_quant_type must be a string')
        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise ValueError('bnb_4bit_use_double_quant must be a boolean')
        if self.load_in_4bit and (not version.parse(importlib.metadata.version('bitsandbytes')) >= version.parse('0.39.0')):
            raise ValueError('4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version')

    def is_quantizable(self):
        """
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        """
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return 'llm_int8'
        elif self.load_in_4bit and self.bnb_4bit_quant_type == 'fp4':
            return 'fp4'
        elif self.load_in_4bit and self.bnb_4bit_quant_type == 'nf4':
            return 'nf4'
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output['bnb_4bit_compute_dtype'] = str(output['bnb_4bit_compute_dtype']).split('.')[1]
        output['load_in_4bit'] = self.load_in_4bit
        output['load_in_8bit'] = self.load_in_8bit
        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f'{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n'

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()
        default_config_dict = BitsAndBytesConfig().to_dict()
        serializable_config_dict = {}
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value
        return serializable_config_dict