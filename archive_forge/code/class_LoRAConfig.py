from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
@dataclass
class LoRAConfig:
    max_lora_rank: int
    max_loras: int
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[torch.dtype] = None
    lora_extra_vocab_size: int = 256
    lora_vocab_padding_size: ClassVar[int] = 256

    def __post_init__(self):
        possible_max_ranks = (8, 16, 32, 64)
        possible_lora_extra_vocab_size = (0, 256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(f'max_lora_rank ({self.max_lora_rank}) must be one of {possible_max_ranks}.')
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(f'lora_extra_vocab_size ({self.lora_extra_vocab_size}) must be one of {possible_lora_extra_vocab_size}.')
        if self.max_loras < 1:
            raise ValueError(f'max_loras ({self.max_loras}) must be >= 1.')
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(f'max_cpu_loras ({self.max_cpu_loras}) must be >= max_loras ({self.max_loras})')

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, 'auto'):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.quantization is not None:
            raise ValueError('LoRA is not supported with quantized models yet.')

    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        if scheduler_config.max_num_batched_tokens > 65528:
            raise ValueError('Due to limitations of the custom LoRA CUDA kernel, max_num_batched_tokens must be <= 65528 when LoRA is enabled.')