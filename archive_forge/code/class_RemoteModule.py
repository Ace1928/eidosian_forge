import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
class RemoteModule(_RemoteModule):
    """
        A RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.

        It generates two methods ``forward_async`` and ``forward`` based on the
        signature of the ``forward`` method of ``module_cls``. ``forward_async``
        runs asynchronously and returns a Future. The arguments of ``forward_async``
        and ``forward`` are the same as the ``forward`` method of the module
        returned by the ``module_cls``.

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature: ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods with the signatures:

        | ``def forward(input: Tensor) -> Tensor:``
        | ``def forward_async(input: Tensor) -> Future[Tensor]:``

    Args:
        remote_device (str): Device on the destination worker where we'd like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".
        module_cls (nn.Module): Class for the module to be created remotely. For example,

            >>> class MyModule(nn.Module):
            >>>     def forward(input):
            >>>         return input + 1
            >>>
            >>> module_cls = MyModule

        args (Sequence, optional): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_cls``, it has a blocking ``forward`` method and an
        asynchronous ``forward_async`` method that returns a future of the ``forward`` call
        on the user-provided module on the remote side.

    Example::
        Run the following code in two different processes:

        >>> # xdoctest: +SKIP("distributed")
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch import nn, Tensor
        >>> from torch.distributed.nn.api.remote_module import RemoteModule
        >>>
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> remote_linear_module = RemoteModule(
        >>>     "worker1/cpu", nn.Linear, args=(20, 30),
        >>> )
        >>> input = torch.randn(128, 20)
        >>> ret_fut = remote_linear_module.forward_async(input)
        >>> ret = ret_fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>>
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Furthermore, a more practical example that is combined with
        `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)
        can be found in this `tutorial <https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>`__.
    """

    def __init__(self, remote_device: str, module_cls: Type[nn.Module], args: Optional[Tuple]=None, kwargs: Optional[Dict[str, Any]]=None):
        super().__init__(remote_device, module_cls, args, kwargs)