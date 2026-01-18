import warnings
from typing import Dict, Optional, Union
from ..models.auto.configuration_auto import AutoConfig
from ..utils.quantization_config import (
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
class AutoQuantizationConfig:
    """
    The Auto-HF quantization config class that takes care of automatically dispatching to the correct
    quantization config given a quantization config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, quantization_config_dict: Dict):
        quant_method = quantization_config_dict.get('quant_method', None)
        if quantization_config_dict.get('load_in_8bit', False) or quantization_config_dict.get('load_in_4bit', False):
            suffix = '_4bit' if quantization_config_dict.get('load_in_4bit', False) else '_8bit'
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError("The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized")
        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            raise ValueError(f'Unknown quantization type, got {quant_method} - supported types are: {list(AUTO_QUANTIZER_MAPPING.keys())}')
        target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
        return target_cls.from_dict(quantization_config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(model_config, 'quantization_config', None) is None:
            raise ValueError(f'Did not found a `quantization_config` in {pretrained_model_name_or_path}. Make sure that the model is correctly quantized.')
        quantization_config_dict = model_config.quantization_config
        quantization_config = cls.from_dict(quantization_config_dict)
        quantization_config.update(kwargs)
        return quantization_config