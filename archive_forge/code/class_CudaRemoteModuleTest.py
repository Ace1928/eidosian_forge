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
class CudaRemoteModuleTest(CommonRemoteModuleTest):

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_valid_device(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = dist_utils.worker_name(dst_rank)
        for remote_module in self._create_remote_module_iter(f'{dst_worker_name}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR]):
            device = rpc.rpc_sync(dst_worker_name, remote_device, (remote_module.module_rref,))
            self.assertEqual(device.type, 'cuda')
            self.assertEqual(device.index, 0)
        for remote_module in self._create_remote_module_iter(f'rank:{dst_rank}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR]):
            device = rpc.rpc_sync(dst_worker_name, remote_device, (remote_module.module_rref,))
            self.assertEqual(device.type, 'cuda')
            self.assertEqual(device.index, 0)

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_invalid_devices(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Expected one of .+ device type at start of device string'):
            [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/foo', modes=[ModuleCreationMode.MODULE_CTOR])]
        with self.assertRaisesRegex(RuntimeError, 'CUDA error: invalid device ordinal'):
            [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/cuda:100', modes=[ModuleCreationMode.MODULE_CTOR])]
        with self.assertRaisesRegex(RuntimeError, "Invalid device string: 'cpu2'"):
            [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/cpu2', modes=[ModuleCreationMode.MODULE_CTOR])]
        with self.assertRaisesRegex(RuntimeError, 'Device string must not be empty'):
            [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/', modes=[ModuleCreationMode.MODULE_CTOR])]
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device: worker1/cuda:0/cuda:1. The valid format is '<workername>/<device>'"):
            [m.forward() for m in self._create_remote_module_iter(f'{dst_worker_name}/cuda:0/cuda:1', modes=[ModuleCreationMode.MODULE_CTOR])]
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device: /. The valid format is '<workername>/<device>'"):
            [m.forward() for m in self._create_remote_module_iter('/', modes=[ModuleCreationMode.MODULE_CTOR])]
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device: /cuda:0. The valid format is '<workername>/<device>'"):
            [m.forward() for m in self._create_remote_module_iter('/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR])]

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_input_moved_to_cuda_device(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        t1 = torch.ones(1)
        args = (t1, 2)
        t2 = t1 * 2
        kwargs = dict(word=t2)
        for remote_module in self._create_remote_module_iter(f'{dst_worker_name}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR]):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + (t2,))))
            self.assertEqual(ret[0].device.type, 'cpu')
            self.assertEqual(ret[2].device.type, 'cpu')
            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + (t2,))))
            self.assertEqual(ret[0].device.type, 'cpu')
            self.assertEqual(ret[2].device.type, 'cpu')

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_input_moved_to_cuda_device_script(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        scripted_remote_module = next(self._create_remote_module_iter(f'{dst_worker_name}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]))

        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            ret = scripted_remote_module.forward(torch.ones(1), 2, '3')
            return ret
        ret = run_forward(scripted_remote_module)
        self.assertEqual(ret, ('3', 2, torch.ones(1)))
        self.assertEqual(ret[2].device.type, 'cpu')