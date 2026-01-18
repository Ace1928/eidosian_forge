import enum
from typing import Tuple
import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import Tensor, nn
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import _REMOTE_MODULE_PICKLED_ATTRIBUTES
from torch.distributed.nn.api.remote_module import _RemoteModule
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
class CommonRemoteModuleTest(RpcAgentTestFixture):

    @property
    def world_size(self):
        return 2

    @staticmethod
    def _create_remote_module_iter(remote_device, modes=None):
        if modes is None:
            modes = ModuleCreationMode.__members__.values()
        args = (1,)
        kwargs = dict(first_kwarg=2)
        if ModuleCreationMode.MODULE_CTOR in modes:
            remote_module = RemoteModule(remote_device, MyModule, args, kwargs)
            yield remote_module
        if ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE in modes:
            remote_module = _RemoteModule(remote_device, create_scripted_module, args, kwargs, _module_interface_cls=MyModuleInterface)
            scripted_remote_module = torch.jit.script(remote_module)
            yield scripted_remote_module