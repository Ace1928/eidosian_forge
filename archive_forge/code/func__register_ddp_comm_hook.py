import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sized, Union, cast
import torch
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler, Sampler
from typing_extensions import Self, override
from lightning_fabric.utilities.distributed import _DatasetSamplerWrapper
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info
from pytorch_lightning.utilities.types import _SizedIterable
def _register_ddp_comm_hook(model: DistributedDataParallel, ddp_comm_state: Optional[object]=None, ddp_comm_hook: Optional[Callable]=None, ddp_comm_wrapper: Optional[Callable]=None) -> None:
    """Function to register communication hook for DDP model https://pytorch.org/docs/master/ddp_comm_hooks.html.

    Args:
        model:
            DDP model
        ddp_comm_state:
            state is passed to the hook and can be used to maintain
            and update any state information that users would like to
            maintain as part of the training process. Examples: error
            feedback in gradient compression, peers to communicate with
            next in GossipGrad etc.
        ddp_comm_hook:
            hook(state: object, bucket: dist._GradBucket) -> torch.futures.Future

            This callable function is called once the bucket is ready. The
            hook can perform whatever processing is needed and return
            a Future indicating completion of any async work (ex: allreduce).
            If the hook doesn't perform any communication, it can also
            just return a completed Future. The Future should hold the
            new value of grad bucket's tensors. Once a bucket is ready,
            c10d reducer would call this hook and use the tensors returned
            by the Future and copy grads to individual parameters.
        ddp_comm_wrapper:
            communication hook wrapper to support a communication hook such
            as FP16 compression as wrapper, which could be combined with
            ddp_comm_hook

    Examples::

        from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks as default,
            powerSGD_hook as powerSGD,
            post_localSGD_hook as post_localSGD,
        )

        # fp16_compress_hook for compress gradients
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_hook=default.fp16_compress_hook,
        )

        # powerSGD_hook
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
        )

        # post_localSGD_hook
        subgroup, _ = torch.distributed.new_subgroups()
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            state=post_localSGD.PostLocalSGDState(
                process_group=None,
                subgroup=subgroup,
                start_localSGD_iter=1_000,
            ),
            ddp_comm_hook=post_localSGD.post_localSGD_hook,
        )

        # fp16_compress_wrapper combined with other communication hook
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
            ddp_comm_wrapper=default.fp16_compress_wrapper,
        )

    """
    if ddp_comm_hook is None:
        return
    ddp_comm_hook: Callable = ddp_comm_hook
    if ddp_comm_wrapper is not None:
        rank_zero_info(f'DDP comm wrapper is provided, apply {ddp_comm_wrapper.__qualname__}({ddp_comm_hook.__qualname__}).')
        ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)
    rank_zero_debug(f'Registering DDP comm hook: {ddp_comm_hook.__qualname__}.')
    model.register_comm_hook(state=ddp_comm_state, hook=ddp_comm_hook)