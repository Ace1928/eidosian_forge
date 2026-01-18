from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
class DeviceConfig:

    def __init__(self, device: str='auto') -> None:
        if device == 'auto':
            if torch.cuda.is_available():
                self.device_type = 'cuda'
            elif is_neuron():
                self.device_type = 'neuron'
            else:
                raise RuntimeError('No supported device detected.')
        else:
            self.device_type = device
        if self.device_type in ['neuron']:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device_type)

    @property
    def is_neuron(self):
        return self.device_type == 'neuron'