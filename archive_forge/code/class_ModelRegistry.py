import importlib
from typing import List, Optional, Type
import torch.nn as nn
from vllm.logger import init_logger
from vllm.utils import is_hip, is_neuron
class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None
        if is_hip():
            if model_arch in _ROCM_UNSUPPORTED_MODELS:
                raise ValueError(f'Model architecture {model_arch} is not supported by ROCm for now.')
            if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(f'Model architecture {model_arch} is partially supported by ROCm: ' + _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch])
        elif is_neuron():
            if model_arch not in _NEURON_SUPPORTED_MODELS:
                raise ValueError(f'Model architecture {model_arch} is not supported by Neuron for now.')
        module_name, model_cls_name = _MODELS[model_arch]
        if is_neuron():
            module_name = _NEURON_SUPPORTED_MODELS[model_arch]
        module = importlib.import_module(f'vllm.model_executor.models.{module_name}')
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())