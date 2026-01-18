import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
class TensorPipeAgentCudaRpcTest(RpcAgentTestFixture, RpcTestCommon):

    def _test_device_maps(self, options, errMsg):
        with self.assertRaisesRegex(ValueError, errMsg):
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        self.assertFalse(rpc.api._is_current_rpc_agent_set())

    @skip_if_lt_x_gpu(2)
    def test_device_maps_wrong_worker_name(self):
        options = self.rpc_backend_options
        options.set_device_map('none_exist', {0: 1})
        self._test_device_maps(options, errMsg='Node worker0 has invalid target node names in its device maps')

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_local_device(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {torch.cuda.device_count(): 0})
        self._test_device_maps(options, errMsg='Node worker0 has source devices with invalid indices in its device map for worker1')

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_max_remote_device(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {0: torch.cuda.device_count()})
        self._test_device_maps(options, errMsg='Node worker0 has target devices with invalid indices in its device map for worker1')

    @skip_if_lt_x_gpu(2)
    def test_device_maps_many_to_one(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {1: 0})
        options.set_device_map(dst, {0: 0})
        self._test_device_maps(options, errMsg='Node worker0 has duplicated target devices in its device map for worker1')

    @skip_if_lt_x_gpu(2)
    def test_device_maps_one_to_many(self):
        if self.rank == 0:
            options = self.rpc_backend_options
            dst = worker_name((self.rank + 1) % self.world_size)
            options.set_device_map(dst, {0: 1})
            with self.assertRaisesRegex(ValueError, '`set_device_map` only supports 1-to-1 mapping'):
                options.set_device_map(dst, {0: 0})

    @skip_if_lt_x_gpu(1)
    def test_device_maps_invalid_min_device(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Device index must not be negative'):
            options.set_device_map(dst, {-1: 0})
        with self.assertRaisesRegex(RuntimeError, 'Device index must not be negative'):
            options.set_device_map(dst, {0: -1})

    @staticmethod
    def _gpu_add(x, y):
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 1]):
            return (x + y).to(0)
        else:
            raise ValueError('Wrong device affinity')

    @skip_if_lt_x_gpu(2)
    def test_device_maps_gpu(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {0: 1, 1: 0})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        ret = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._gpu_add, args=(torch.zeros(2).to(0), torch.ones(2).to(0)))
        self.assertEqual(ret.device, torch.device(1))
        self.assertEqual(ret, (torch.zeros(2) + torch.ones(2)).to(1))
        rpc.shutdown()

    @staticmethod
    def _gpu_add_given_devices(x, y, x_to, y_to, z_to):
        x_device = 'cpu' if x.device.type == 'cpu' else x.device.index
        y_device = 'cpu' if y.device.type == 'cpu' else y.device.index
        if x_device == x_to and y_device == y_to:
            return x.to(z_to) + y.to(z_to)
        else:
            raise ValueError('Wrong device affinity')

    def _test_device_maps_gpu(self, x_from, y_from, z_to, device_map, dst=None, fn=None):
        fn = TensorPipeAgentCudaRpcTest._gpu_add_given_devices if fn is None else fn
        x_to = device_map[x_from]
        y_to = device_map[y_from]
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size) if dst is None else dst
        options.set_device_map(dst, device_map)
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        x = torch.zeros(2).to(x_from)
        y = torch.ones(2).to(y_from)
        ret = rpc.rpc_sync(dst, fn, args=(x, y, x_to, y_to, z_to))
        reverse_device_map = {device_map[k]: k for k in device_map}
        z_from = reverse_device_map[z_to]
        ret_device = 'cpu' if ret.device.type == 'cpu' else ret.device.index
        self.assertEqual(ret_device, z_from)
        self.assertEqual(ret, torch.ones(2).to(z_from))
        rpc.shutdown()

    def test_device_map_cpu(self):
        self._test_device_maps_gpu(x_from='cpu', y_from='cpu', z_to='cpu', device_map={'cpu': 'cpu'}, fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices)

    @skip_if_lt_x_gpu(1)
    def test_device_map_cpu_to_gpu_default(self):
        self._test_device_maps_gpu(x_from='cpu', y_from='cpu', z_to=0, device_map={'cpu': 0}, fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices)

    @skip_if_lt_x_gpu(2)
    def test_device_map_cpu_to_gpu_non_default(self):
        self._test_device_maps_gpu(x_from='cpu', y_from='cpu', z_to=1, device_map={'cpu': 1}, fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices)

    @skip_if_lt_x_gpu(1)
    def test_device_map_gpu_to_cpu_default(self):
        self._test_device_maps_gpu(x_from=0, y_from=0, z_to='cpu', device_map={0: 'cpu'}, fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices)

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_to_cpu_non_default(self):
        self._test_device_maps_gpu(x_from=1, y_from=1, z_to='cpu', device_map={1: 'cpu'}, fn=TensorPipeAgentCudaRpcTest._gpu_add_given_devices)

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_default(self):
        self._test_device_maps_gpu(x_from=0, y_from=0, z_to=0, device_map={0: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_non_default(self):
        self._test_device_maps_gpu(x_from=1, y_from=1, z_to=1, device_map={1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_default_to_non_default(self):
        self._test_device_maps_gpu(x_from=0, y_from=0, z_to=1, device_map={0: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_non_default_to_default(self):
        self._test_device_maps_gpu(x_from=1, y_from=1, z_to=0, device_map={1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_1(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=0, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_2(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=1, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_3(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=0, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_4(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=1, device_map={0: 0, 1: 1})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_5(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=0, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_6(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=1, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_7(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=0, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_8(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=1, device_map={0: 1, 1: 0})

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_1(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=0, device_map={0: 0, 1: 1}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_2(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=1, device_map={0: 0, 1: 1}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_3(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=0, device_map={0: 0, 1: 1}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_4(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=1, device_map={0: 0, 1: 1}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_5(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=0, device_map={0: 1, 1: 0}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_6(self):
        self._test_device_maps_gpu(x_from=0, y_from=1, z_to=1, device_map={0: 1, 1: 0}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_7(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=0, device_map={0: 1, 1: 0}, dst=worker_name(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_device_map_gpu_mixed_self_8(self):
        self._test_device_maps_gpu(x_from=1, y_from=0, z_to=1, device_map={0: 1, 1: 0}, dst=worker_name(self.rank))

    @staticmethod
    def _gpu_add_multi_gpu(x, y):
        if all([x.is_cuda, x.device.index == 1, y.is_cuda, y.device.index == 0]):
            return (x.to(0) + y, x - y.to(1))
        else:
            raise ValueError('Wrong device affinity')

    def _test_device_maps_multi_gpu(self, dst):
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 0})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        x = torch.zeros(2).to(0)
        y = torch.ones(2).to(1)
        rets = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._gpu_add_multi_gpu, args=(x, y))
        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_device_maps_multi_gpu(dst)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_multi_gpu_self(self):
        dst = worker_name(self.rank)
        self._test_device_maps_multi_gpu(dst)

    @staticmethod
    def _gpu_add_return_to_gpu(x, y):
        if x.device.type == 'cpu' and y.device.type == 'cpu':
            return ((x + y).to(0), (x - y).to(1), (x * y).to(2), (x / y).to(3))
        else:
            raise ValueError('Wrong device affinity')

    @skip_if_lt_x_gpu(2)
    def test_device_maps_in_options(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=options.init_method, num_worker_threads=options.num_worker_threads, device_maps={dst: {0: 1, 1: 0}}, _transports=tp_transports()))
        rets = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._gpu_add_multi_gpu, args=(torch.zeros(2).to(0), torch.ones(2).to(1)))
        self.assertEqual(rets[0].device, torch.device(1))
        self.assertEqual(rets[1].device, torch.device(0))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(1))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        rpc.shutdown()

    def _test_device_maps_return_to_gpu(self, dst):
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 1})
        options.set_device_map(dst, {1: 2})
        options.set_device_map(dst, {2: 3})
        options.set_device_map(dst, {3: 0})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        rets = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._gpu_add_return_to_gpu, args=(torch.zeros(2), torch.ones(2)))
        for i in range(len(rets)):
            self.assertEqual(rets[i].device, torch.device((3 + i) % 4))
        self.assertEqual(rets[0], (torch.zeros(2) + torch.ones(2)).to(3))
        self.assertEqual(rets[1], (torch.zeros(2) - torch.ones(2)).to(0))
        self.assertEqual(rets[2], (torch.zeros(2) * torch.ones(2)).to(1))
        self.assertEqual(rets[3], (torch.zeros(2) / torch.ones(2)).to(2))
        rpc.shutdown()

    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._test_device_maps_return_to_gpu(dst)

    @skip_if_lt_x_gpu(4)
    def test_device_maps_return_to_gpu_self(self):
        dst = worker_name(self.rank)
        self._test_device_maps_return_to_gpu(dst)

    @staticmethod
    def _add_to_gpu(x, y):
        return (x + y).to(0)

    def _test_device_maps_missing_config(self, mode):
        dst = worker_name((self.rank + 1) % self.world_size)
        errMsg = 'TensorPipe RPC backend only supports CPU tensors by default.*`set_device_map` on `TensorPipeRpcBackendOptions`'
        with self.assertRaisesRegex(RuntimeError, errMsg):
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(dst, torch.add, args=(torch.zeros(2).to(0), 1))
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(dst, torch.add, args=(torch.zeros(2).to(0), 1)).to_here()
            else:
                raise ValueError(f'unexpected mode {mode}')
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        self.assertEqual(ret, torch.ones(2) + 1)

    def _test_device_maps_missing_config_response(self, mode):
        dst = worker_name((self.rank + 1) % self.world_size)
        errMsg = 'Response device mapping is not available'
        with self.assertRaisesRegex(RuntimeError, errMsg):
            if mode == RPCExecMode.SYNC:
                rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._add_to_gpu, args=(torch.zeros(2), 1))
            elif mode == RPCExecMode.REMOTE:
                rpc.remote(dst, TensorPipeAgentCudaRpcTest._add_to_gpu, args=(torch.zeros(2), 1)).to_here()
            else:
                raise ValueError(f'unexpected mode {mode}')
        ret = rpc.rpc_sync(dst, torch.add, args=(torch.ones(2), 1))
        self.assertEqual(ret, torch.ones(2) + 1)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config(self):
        self._test_device_maps_missing_config(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    def test_device_maps_missing_config_not_timeout(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        timeout = rpc.get_rpc_timeout()
        tik = time.time()
        self._test_device_maps_missing_config(RPCExecMode.SYNC)
        rpc.shutdown()
        tok = time.time()
        self.assertTrue(tok - tik < timeout)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_loop(self):
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response(self):
        self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_response_loop(self):
        for _ in range(self.rpc_backend_options.num_worker_threads + 5):
            self._test_device_maps_missing_config_response(RPCExecMode.SYNC)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_remote(self):
        self._test_device_maps_missing_config(RPCExecMode.REMOTE)

    @skip_if_lt_x_gpu(1)
    @dist_init
    def test_device_maps_missing_config_remote_response(self):
        self._test_device_maps_missing_config_response(RPCExecMode.REMOTE)

    @skip_if_lt_x_gpu(2)
    def test_device_maps_remote(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {1: 0})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        rref = rpc.remote(dst, TensorPipeAgentCudaRpcTest._add_to_gpu, args=(torch.zeros(2), 1))
        self.assertEqual(rref.to_here().device.index, 1)
        self.assertEqual(rref.to_here(), torch.ones(2).to(1))
        rpc.shutdown()

    @staticmethod
    def _slow_add_on_user_stream(x, y):
        s0 = torch.cuda.current_stream(x.device)
        s1 = torch.cuda.Stream(device=x.device)
        s1.wait_stream(s0)
        x.record_stream(s1)
        y.record_stream(s1)
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            z = x + y
        s0.wait_stream(s1)
        z.record_stream(s0)
        return z

    def _test_custom_stream(self, fn, device_map):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, device_map)
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        fn(dst)
        rpc.shutdown()

    def _test_stream_sync(self, dst):
        x = torch.ones(2, 2).to(0)
        ret = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._slow_add_on_user_stream, args=(x, x))
        self.assertEqual(ret, 2 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream(self):
        self._test_custom_stream(self._test_stream_sync, {'cuda:0': 'cuda:1'})

    def _test_stream_multi_async(self, dst):
        futs = []
        for i in range(20):
            x = torch.ones(2, 2).to(0) * i
            futs.append(rpc.rpc_async(dst, TensorPipeAgentCudaRpcTest._slow_add_on_user_stream, args=(x, x)))
        for i in range(20):
            self.assertEqual(futs[i].wait(), 2 * torch.ones(2, 2).to(0) * i)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_multi(self):
        self._test_custom_stream(self._test_stream_multi_async, {'cuda:0': 'cuda:1'})

    @staticmethod
    def _nested_slow_add_on_user_stream(dst, x, y, z):
        ret = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._slow_add_on_user_stream, args=(x, y))
        return TensorPipeAgentCudaRpcTest._slow_add_on_user_stream(ret, z)

    def _test_stream_nested_sync(self, dst):
        x = torch.ones(2, 2).to(0)
        y = torch.ones(2, 2).to(0) * 2
        z = torch.ones(2, 2).to(0) * 3
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream, args=(nested_dst, x, y, z))
        self.assertEqual(ret, 6 * x)

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested(self):
        self._test_custom_stream(self._test_stream_nested_sync, {'cuda:0': 'cuda:1', 'cuda:1': 'cuda:0'})

    def _test_stream_nested_multi_async(self, dst):
        if self.rank == 0:
            futs = []
            n = 5
            xs, ys, zs = ([], [], [])
            for i in range(n):
                x = torch.ones(2, 2).to(0) * (i - 1)
                y = torch.ones(2, 2).to(0) * i
                z = torch.ones(2, 2).to(0) * (i + 1)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                nested_dst = worker_name((self.rank + 2) % self.world_size)
                futs.append(rpc.rpc_async(dst, TensorPipeAgentCudaRpcTest._nested_slow_add_on_user_stream, args=(nested_dst, x, y, z)))
            for i in range(n):
                self.assertEqual(futs[i].wait(), xs[i] + ys[i] + zs[i])

    @skip_if_lt_x_gpu(2)
    def test_custom_stream_nested_multi(self):
        self._test_custom_stream(self._test_stream_nested_multi_async, {'cuda:0': 'cuda:1', 'cuda:1': 'cuda:0'})

    @staticmethod
    def _gpu_add_wrong_gpus(x, y):
        if x.is_cuda and y.is_cuda:
            return x.cpu() + y.cuda()
        else:
            raise ValueError('Wrong device affinity')

    @skip_if_lt_x_gpu(1)
    def test_device_mismatch(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 0})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        x = torch.zeros(2).to(0)
        y = torch.ones(2).to(0)
        with self.assertRaisesRegex(RuntimeError, 'Expected all tensors to be on the same device, but found at least two devices'):
            rets = rpc.rpc_sync(dst, TensorPipeAgentCudaRpcTest._gpu_add_wrong_gpus, args=(x, y))
        rpc.shutdown()

    def _test_rref_synchronization(self, local_device, remote_device):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {local_device: remote_device})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        if self.rank == 1:
            rref = rpc.remote(dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                x = torch.randn(200, 1, 28, 28).to(local_device)
                actual = rref.remote().forward(x).to_here()
                expected = rref.rpc_sync().forward(x)
                self.assertEqual(actual, expected)
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_to_here_synchronization1(self):
        self._test_rref_synchronization('cuda:0', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization2(self):
        self._test_rref_synchronization('cuda:1', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization3(self):
        self._test_rref_synchronization('cuda:1', 'cuda:1')

    @skip_if_lt_x_gpu(2)
    def test_rref_to_here_synchronization4(self):
        self._test_rref_synchronization('cuda:0', 'cuda:1')

    def _test_rref_as_arg_synchronization(self, local_device, remote_device, devicesOptions=None):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {local_device: remote_device})
        input_src = worker_name((self.rank - 1 + self.world_size) % self.world_size)
        options.set_device_map(input_src, {remote_device: local_device})
        if devicesOptions is not None:
            options.set_devices(devicesOptions[self.rank])
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        if self.rank == 1:
            rref = rpc.remote(dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                rref_x = RRef(torch.randn(200, 1, 28, 28).to(local_device))
                actual = rref.remote().forward(rref_x, True).to_here()
                expected = rref.rpc_sync().forward(rref_x, True)
                self.assertEqual(actual, expected)
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_as_arg_synchronization1(self):
        self._test_rref_as_arg_synchronization('cuda:0', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization2(self):
        self._test_rref_as_arg_synchronization('cuda:1', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization3(self):
        self._test_rref_as_arg_synchronization('cuda:1', 'cuda:1')

    @skip_if_lt_x_gpu(2)
    def test_rref_as_arg_synchronization4(self):
        self._test_rref_as_arg_synchronization('cuda:0', 'cuda:1')

    @skip_if_lt_x_gpu(1)
    def test_rref_as_arg_synchronization5(self):
        self._test_rref_as_arg_synchronization('cuda:0', 'cuda:0', [['cuda:0'] for _ in range(4)])

    @staticmethod
    def _rref_relay(rref):
        return rref.to_here()

    def _test_rref_forward_synchronization(self, local_device, remote_device):
        options = self.rpc_backend_options
        input_src = worker_name(0)
        model_dst = worker_name(1)
        out_relay = worker_name(2)
        if self.rank == 0:
            options.set_device_map(model_dst, {local_device: remote_device})
            options.set_device_map(out_relay, {local_device: local_device})
        elif self.rank == 1:
            options.set_device_map(input_src, {remote_device: local_device})
        elif self.rank == 2:
            options.set_device_map(model_dst, {local_device: remote_device})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        if self.rank == 0:
            rref = rpc.remote(model_dst, MyConvNetForMNIST, args=(remote_device,))
            for _ in range(10):
                rref_input = RRef(torch.randn(200, 1, 28, 28).to(local_device))
                rref_out = rref.remote().forward(rref_input, True)
                out = rpc.remote(out_relay, TensorPipeAgentCudaRpcTest._rref_relay, args=(rref_out,)).to_here()
                expected = rref.rpc_sync().forward(rref_input, True)
                self.assertEqual(out, expected)
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_rref_forward_synchronization1(self):
        self._test_rref_forward_synchronization('cuda:0', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization2(self):
        self._test_rref_forward_synchronization('cuda:0', 'cuda:1')

    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization3(self):
        self._test_rref_forward_synchronization('cuda:1', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_rref_forward_synchronization4(self):
        self._test_rref_forward_synchronization('cuda:1', 'cuda:1')

    def _test_owner_rref_forward_synchronization(self, local_device, remote_device):
        if self.rank == 0:
            options = self.rpc_backend_options
            options.set_device_map('w0', {local_device: remote_device})
            rpc.init_rpc('w0', rank=0, world_size=1, rpc_backend_options=options)
            model = rpc.remote('w0', torch.nn.Linear, (2048, 20000)).remote().to(remote_device)
            for _ in range(30):
                data = torch.rand(2048, 2048).to(local_device)
                output = model.rpc_sync().forward(data)
                v0 = rpc.RRef(output).remote().sum().to_here().item()
                v1 = output.sum().item()
                self.assertEqual(v0, v1)
            rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_owner_rref_forward_synchronization1(self):
        self._test_owner_rref_forward_synchronization('cuda:0', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization2(self):
        self._test_owner_rref_forward_synchronization('cuda:0', 'cuda:1')

    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization3(self):
        self._test_owner_rref_forward_synchronization('cuda:1', 'cuda:0')

    @skip_if_lt_x_gpu(2)
    def test_owner_rref_forward_synchronization4(self):
        self._test_owner_rref_forward_synchronization('cuda:1', 'cuda:1')

    @staticmethod
    def _return_tensor_view(i):
        x = torch.ones(1000, 200).cuda(0) * i
        torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
        return x.split(100)[0]

    @skip_if_lt_x_gpu(1)
    def test_tensor_view_as_return_value(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {0: 0})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        futs = []
        for i in range(5):
            futs.append(rpc.rpc_async(dst, TensorPipeAgentCudaRpcTest._return_tensor_view, args=(i,)))
        for i in range(5):
            self.assertEqual(torch.ones(100, 200) * i, futs[i].wait())
        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_devices_option_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Node worker0 has unexpected source devices in its device map for worker1'):
            dst = worker_name((self.rank + 1) % self.world_size)
            options = self.rpc_backend_options
            options.set_device_map(dst, {0: 0})
            options.set_devices([1])
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
            rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_devices_option_mismatch_reverse(self):
        with self.assertRaisesRegex(ValueError, 'Node worker0 has unexpected target devices in its device map for worker1'):
            dst = worker_name((self.rank + 1) % self.world_size)
            options = rpc.TensorPipeRpcBackendOptions(init_method=self.rpc_backend_options.init_method, num_worker_threads=self.rpc_backend_options.num_worker_threads, device_maps={dst: {0: 1}}, devices=[0])
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
            rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_int(self):
        fut = Future(devices=[0])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_str(self):
        fut = Future(devices=['cuda:0'])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_as_device(self):
        fut = Future(devices=[torch.device('cuda', 0)])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_device_not_cuda(self):
        with self.assertRaisesRegex(ValueError, 'Expected devices to have indices, got cpu'):
            fut = Future(devices=['cpu'])

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_cuda_tensor(self):
        self._test_cuda_future_extraction(wrapper=lambda t: t, unwrapper=lambda v: v, sparse_tensor=False)

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_list_with_cuda_tensor(self):
        self._test_cuda_future_extraction(wrapper=lambda t: [t], unwrapper=lambda v: v[0], sparse_tensor=False)

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_custom_class_with_cuda_tensor(self):
        self._test_cuda_future_extraction(wrapper=TensorWrapper, unwrapper=lambda v: v.tensor, sparse_tensor=False)

    @skip_if_lt_x_gpu(2)
    def test_cuda_future_callback_changes_devices(self):
        tensor0 = torch.zeros((100,), device='cuda:0')
        tensor1 = torch.zeros((100,), device='cuda:1')
        parent_future = Future(devices=['cuda:0', 'cuda:1'])

        def cb(fut):
            t0 = fut.value()
            tensor1.copy_(t0, non_blocking=True)
            return tensor1
        child_future = parent_future.then(cb)
        with torch.cuda.device('cuda:0'):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor0.fill_(1)
                parent_future.set_result(tensor0)
        with torch.cuda.device('cuda:1'):
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(another_stream):
                self.assertTrue(torch.eq(child_future.wait(), 1).all().item())

    @skip_if_lt_x_gpu(2)
    def test_cuda_future_value_on_bad_device(self):
        tensor0 = torch.zeros((100,), device='cuda:0')
        tensor1 = torch.zeros((100,), device='cuda:1')
        parent_future = Future(devices=['cuda:1'])

        def cb(fut):
            with torch.cuda.device('cuda:1'):
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor1.fill_(1)
                return tensor1
        child_future = parent_future.then(cb)
        with torch.cuda.device('cuda:0'):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor0.fill_(1)
                parent_future.set_result(tensor0)
        with self.assertRaisesRegex(ValueError, 'The result contained tensors residing on device\\(s\\) cuda:0 which are not among the expected device\\(s\\) cuda:1'):
            parent_future.wait()
        with torch.cuda.device('cuda:1'):
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(another_stream):
                self.assertTrue(torch.eq(child_future.wait(), 1).all().item())

    @skip_if_lt_x_gpu(1)
    def test_async_execution_with_cuda_future(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {'cuda:0': 'cuda:0'})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        t = torch.zeros((100,), device='cuda:0')
        fut = rpc.rpc_async(dst, async_cuda_sleep_and_set_to_one, args=(t,))
        another_stream = torch.cuda.Stream('cuda:0')
        with torch.cuda.stream(another_stream):
            self.assertTrue(torch.eq(fut.wait(), 1).all().item())
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_async_execution_nested_with_cuda_future(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        nested_dst = worker_name((self.rank + 2) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {'cuda:0': 'cuda:0'})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        a = torch.ones((100,), device='cuda:0')
        b = torch.ones((100,), device='cuda:0')
        c = torch.ones((100,), device='cuda:0')
        fut = rpc.rpc_async(dst, async_cuda_nested_add, args=(nested_dst, a, b, c))
        another_stream = torch.cuda.Stream('cuda:0')
        with torch.cuda.stream(another_stream):
            self.assertTrue(torch.eq(fut.wait(), 3).all().item())
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_modify_tensor_inplace(self):
        tensor = torch.zeros((100,), device='cuda:0')
        future = Future(devices=['cuda:0'])
        future.set_result(tensor)
        tensor.fill_(1)
        future.wait()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_replace_tensor(self):
        tensor_list = [torch.zeros((100,), device='cuda:0')]
        future = Future(devices=['cuda:0'])
        future.set_result(tensor_list)
        tensor_list[0] = torch.ones((100,), device='cuda:0')
        future.wait()

    @skip_if_lt_x_gpu(1)
    def test_rref_with_unpickleable_attributes(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        options = self.rpc_backend_options
        options.set_device_map(dst, {'cuda:0': 'cuda:0'})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        rref = rpc.remote(dst, TensorWrapper, args=(torch.zeros(42, device='cuda:0'),))
        rref.rpc_sync().increase(1)
        ret = rref.rpc_sync().sum()
        self.assertEqual(ret, 42)
        rpc.shutdown()

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_cuda_sparse_tensor(self):
        self._test_cuda_future_extraction(wrapper=lambda t: t, unwrapper=lambda v: v, sparse_tensor=True)

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_list_with_cuda_sparse_tensor(self):
        self._test_cuda_future_extraction(wrapper=lambda t: [t], unwrapper=lambda v: v[0], sparse_tensor=True)

    @skip_if_lt_x_gpu(1)
    def test_cuda_future_can_extract_custom_class_with_cuda_sparse_tensor(self):
        self._test_cuda_future_extraction(wrapper=TensorWrapper, unwrapper=lambda v: v.tensor, sparse_tensor=True)