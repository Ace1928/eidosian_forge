from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _verify_cache_dtype(self) -> None:
    if self.cache_dtype == 'auto':
        pass
    elif self.cache_dtype == 'fp8_e5m2':
        nvcc_cuda_version = get_nvcc_cuda_version()
        if nvcc_cuda_version and nvcc_cuda_version < Version('11.8'):
            raise ValueError('FP8 is not supported when cuda version is lower than 11.8.')
        device_name = torch.cuda.get_device_name()
        if 'AMD' in device_name:
            raise NotImplementedError('FP8_E5M2 KV Cache on AMD GPU has not been supported yet.')
        logger.info('Using fp8_e5m2 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. But it may cause slight accuracy drop. Currently we only support fp8 without scaling factors and make e5m2 as a default format.')
    else:
        raise ValueError(f'Unknown kv cache dtype: {self.cache_dtype}')