import warnings
from typing import Dict, Optional, Union
from ..models.auto.configuration_auto import AutoConfig
from ..utils.quantization_config import (
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_gptq import GptqHfQuantizer

        handles situations where both quantization_config from args and quantization_config from model config are present.
        