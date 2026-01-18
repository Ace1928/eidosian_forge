from enum import Enum
from functools import partial
import torch.distributed as dist
from . import (
class DDPCommHookType(Enum):
    """
    DDPCommHookType enumerates the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    as names and ``ddp_comm_hook_wrapper`` partials with hook specified. As an example,
    you can register allreduce hook by
    ``DDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
    """
    ALLREDUCE = partial(_ddp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
    FP16_COMPRESS = partial(_ddp_comm_hook_wrapper, comm_hook=default.fp16_compress_hook)
    BF16_COMPRESS = partial(_ddp_comm_hook_wrapper, comm_hook=default.bf16_compress_hook)
    QUANTIZE_PER_TENSOR = partial(_ddp_comm_hook_wrapper, comm_hook=quantization.quantization_pertensor_hook)
    QUANTIZE_PER_CHANNEL = partial(_ddp_comm_hook_wrapper, comm_hook=quantization.quantization_perchannel_hook)
    POWER_SGD = partial(_powerSGD_comm_hook_wrapper, comm_hook=powerSGD.powerSGD_hook, matrix_approximation_rank=1)
    POWER_SGD_RANK2 = partial(_powerSGD_comm_hook_wrapper, comm_hook=powerSGD.powerSGD_hook, matrix_approximation_rank=2)
    BATCHED_POWER_SGD = partial(_powerSGD_comm_hook_wrapper, comm_hook=powerSGD.batched_powerSGD_hook, matrix_approximation_rank=1)
    BATCHED_POWER_SGD_RANK2 = partial(_powerSGD_comm_hook_wrapper, comm_hook=powerSGD.batched_powerSGD_hook, matrix_approximation_rank=2)
    NOOP = partial(_ddp_comm_hook_wrapper, comm_hook=debugging.noop_hook)