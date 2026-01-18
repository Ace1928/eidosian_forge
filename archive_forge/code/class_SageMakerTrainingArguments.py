import importlib.util
import json
import os
import warnings
from dataclasses import dataclass, field
import torch
from ..training_args import TrainingArguments
from ..utils import cached_property, is_sagemaker_dp_enabled, logging
@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = field(default='', metadata={'help': 'Used by the SageMaker launcher to send mp-specific args. Ignored in SageMakerTrainer'})

    def __post_init__(self):
        super().__post_init__()
        warnings.warn('`SageMakerTrainingArguments` is deprecated and will be removed in v5 of Transformers. You can use `TrainingArguments` instead.', FutureWarning)

    @cached_property
    def _setup_devices(self) -> 'torch.device':
        logger.info('PyTorch: setting up devices')
        if torch.distributed.is_available() and torch.distributed.is_initialized() and (self.local_rank == -1):
            logger.warning('torch.distributed process group is initialized, but local_rank == -1. In order to use Torch DDP, launch your script with `python -m torch.distributed.launch')
        if self.no_cuda:
            device = torch.device('cpu')
            self._n_gpu = 0
        elif is_sagemaker_model_parallel_available():
            local_rank = smp.local_rank()
            device = torch.device('cuda', local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            import smdistributed.dataparallel.torch.torch_smddp
            torch.distributed.init_process_group(backend='smddp', timeout=self.ddp_timeout_delta)
            self.local_rank = int(os.getenv('SMDATAPARALLEL_LOCAL_RANK'))
            device = torch.device('cuda', self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self._n_gpu = torch.cuda.device_count()
        else:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl', timeout=self.ddp_timeout_delta)
            device = torch.device('cuda', self.local_rank)
            self._n_gpu = 1
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        return device

    @property
    def world_size(self):
        if is_sagemaker_model_parallel_available():
            return smp.dp_size()
        return super().world_size

    @property
    def place_model_on_device(self):
        return not is_sagemaker_model_parallel_available()

    @property
    def _no_sync_in_gradient_accumulation(self):
        return False