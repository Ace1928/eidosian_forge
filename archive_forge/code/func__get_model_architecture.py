from typing import Type
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import ModelConfig, DeviceConfig
from vllm.model_executor.models import ModelRegistry
def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, 'architectures', [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(f'Model architectures {architectures} are not supported for now. Supported architectures: {ModelRegistry.get_supported_archs()}')