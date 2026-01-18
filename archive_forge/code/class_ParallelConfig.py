from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
    """

    def __init__(self, pipeline_parallel_size: int, tensor_parallel_size: int, worker_use_ray: bool, max_parallel_loading_workers: Optional[int]=None, disable_custom_all_reduce: bool=False) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        if is_neuron():
            self.tensor_parallel_size = 1
            self.neuron_tp_degree = tensor_parallel_size
        else:
            self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.world_size = pipeline_parallel_size * self.tensor_parallel_size
        if self.world_size > 1 and (not is_neuron()):
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError('Pipeline parallelism is not supported yet.')
        if not self.disable_custom_all_reduce and self.world_size > 1:
            if is_hip():
                self.disable_custom_all_reduce = True
                logger.info('Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.')
            elif self.pipeline_parallel_size > 1:
                self.disable_custom_all_reduce = True
                logger.info('Disabled the custom all-reduce kernel because it is not supported with pipeline parallelism.')
        if not self.disable_custom_all_reduce and self.world_size > 1:
            self.disable_custom_all_reduce = True
            logger.info('Custom all-reduce kernels are temporarily disabled due to stability issues. We will re-enable them once the issues are resolved.')