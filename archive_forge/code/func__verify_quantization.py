from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _verify_quantization(self) -> None:
    supported_quantization = ['awq', 'gptq', 'squeezellm', 'marlin']
    rocm_not_supported_quantization = ['awq', 'marlin']
    if self.quantization is not None:
        self.quantization = self.quantization.lower()
    hf_quant_config = getattr(self.hf_config, 'quantization_config', None)
    if hf_quant_config is not None:
        hf_quant_method = str(hf_quant_config['quant_method']).lower()
        if hf_quant_method == 'gptq' and 'is_marlin_format' in hf_quant_config and hf_quant_config['is_marlin_format']:
            hf_quant_method = 'marlin'
        if self.quantization is None:
            self.quantization = hf_quant_method
        elif self.quantization != hf_quant_method:
            raise ValueError(f'Quantization method specified in the model config ({hf_quant_method}) does not match the quantization method specified in the `quantization` argument ({self.quantization}).')
    if self.quantization is not None:
        if self.quantization not in supported_quantization:
            raise ValueError(f'Unknown quantization method: {self.quantization}. Must be one of {supported_quantization}.')
        if is_hip() and self.quantization in rocm_not_supported_quantization:
            raise ValueError(f'{self.quantization} quantization is currently not supported in ROCm.')
        if self.quantization != 'marlin':
            logger.warning(f'{self.quantization} quantization is not fully optimized yet. The speed can be slower than non-quantized models.')