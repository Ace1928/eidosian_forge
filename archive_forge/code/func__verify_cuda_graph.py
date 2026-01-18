from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _verify_cuda_graph(self) -> None:
    if self.max_context_len_to_capture is None:
        self.max_context_len_to_capture = self.max_model_len
    self.max_context_len_to_capture = min(self.max_context_len_to_capture, self.max_model_len)