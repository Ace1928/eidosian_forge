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
class RpcTest(RpcAgentTestFixture, RpcTestCommon):

    @dist_init
    def test_worker_id(self):
        n = self.rank + 1
        peer_rank = n % self.world_size
        self_worker_info = rpc.get_worker_info()
        peer_worker_info = rpc.get_worker_info(worker_name(peer_rank))
        self.assertEqual(self_worker_info.name, worker_name(self.rank))
        self.assertEqual(peer_worker_info.name, worker_name(peer_rank))
        with self.assertRaisesRegex(RuntimeError, 'could not find destination'):
            unknown_worker_id = rpc.get_worker_info('WorkerUnknown')

    @dist_init
    def test_get_worker_infos(self):
        worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()
        worker_names = {worker_info.name for worker_info in worker_infos}
        expected_worker_names = {worker_name(rank) for rank in range(self.world_size)}
        self.assertEqual(worker_names, expected_worker_names)
        worker_ids = {worker_info.id for worker_info in worker_infos}
        expected_worker_ids = set(range(self.world_size))
        self.assertEqual(worker_ids, expected_worker_ids)

    @dist_init
    def test_self_add(self):
        self_worker_info = rpc.get_worker_info()
        self_worker_name = worker_name(self.rank)
        fut = rpc.rpc_async(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        ret = rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_send_to_rank(self):
        dst_rank = (self.rank + 1) % self.world_size
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
            self.assertEqual(ret, torch.ones(2, 2) + 1)
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(self.world_size + 1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(-1, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(dst_rank + 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(dst_rank - 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1))

    @dist_init
    def test_self_py_udf_remote(self):
        self._self_py_udf_remote(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_rpc_arg(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_rpc_arg(dst, torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg(self):
        self._self_remote_rref_as_rpc_arg(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_remote_arg(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_remote_arg(dst, torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_self_remote_arg(self):
        self._self_remote_rref_as_remote_arg(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_rref_proxy_non_exist(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))
        msg = "has no attribute 'non_exist'"
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_sync().non_exist()
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_async().non_exist().wait()
        with self.assertRaisesRegex(AttributeError, msg):
            rref.remote().non_exist()

    def _test_rref_proxy_tensor(self, dst):
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))
        expected = torch.ones(2, 2) + 1 + 3
        self.assertEqual(expected.size(), rref.rpc_sync().size())
        self.assertEqual(expected + 1, rref.rpc_async().add(1).wait())
        self.assertEqual(expected.view(1, 4), rref.remote().view(1, 4).to_here())

    @dist_init
    def test_rref_proxy_tensor(self):
        self._test_rref_proxy_tensor(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_tensor_self(self):
        self._test_rref_proxy_tensor(rpc.get_worker_info())

    @dist_init
    def test_rref_proxy_reuse(self):
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), my_function, args=(torch.ones(2, 2), 1, 3))
        expected = torch.ones(2, 2) + 1 + 3
        proxy_rpc_sync = rref.rpc_sync()
        proxy_rpc_async = rref.rpc_async()
        proxy_remote = rref.remote()
        self.assertEqual(expected.size(), proxy_rpc_sync.size())
        self.assertEqual(expected + 1, proxy_rpc_sync.add(1))
        self.assertEqual(expected.view(1, 4), proxy_rpc_sync.view(1, 4))
        self.assertEqual(expected.size(), proxy_rpc_async.size().wait())
        self.assertEqual(expected + 3, proxy_rpc_async.add(3).wait())
        self.assertEqual(expected.view(4, 1), proxy_rpc_async.view(4, 1).wait())
        self.assertEqual(expected.size(), proxy_remote.size().to_here())
        self.assertEqual(expected + 5, proxy_remote.add(5).to_here())
        self.assertEqual(expected.view(-1), proxy_remote.view(-1).to_here())

    def _test_rref_proxy_class(self, dst):
        rref = rpc.remote(dst, MyClass, args=(7,))
        expected = MyClass(7)
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())
        expected.increment_value(3)
        self.assertEqual(None, rref.rpc_sync().increment_value(1))
        self.assertEqual(None, rref.rpc_async().increment_value(1).wait())
        self.assertEqual(None, rref.remote().increment_value(1).to_here())
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())
        self.assertEqual(expected.my_instance_method(2), rref.rpc_sync().my_instance_method(2))
        self.assertEqual(expected.my_instance_method(3), rref.rpc_async().my_instance_method(3).wait())
        self.assertEqual(expected.my_instance_method(4), rref.remote().my_instance_method(4).to_here())
        self.assertEqual(expected.my_static_method(9), rref.rpc_sync().my_static_method(9))
        self.assertEqual(expected.my_static_method(10), rref.rpc_async().my_static_method(10).wait())
        self.assertEqual(expected.my_static_method(11), rref.remote().my_static_method(11).to_here())
        self.assertEqual(expected.my_class_method(2, torch.zeros(2, 2)), rref.rpc_sync().my_class_method(2, torch.zeros(2, 2)))
        self.assertEqual(expected.my_class_method(2, torch.ones(3, 3)), rref.rpc_async().my_class_method(2, torch.ones(3, 3)).wait())
        self.assertEqual(expected.my_class_method(2, torch.ones(4, 4)), rref.remote().my_class_method(2, torch.ones(4, 4)).to_here())

    @dist_init
    def test_rref_proxy_class(self):
        self._test_rref_proxy_class(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_class_self(self):
        self._test_rref_proxy_class(rpc.get_worker_info())

    @mock.patch.object(torch.distributed.autograd, '_init')
    @mock.patch.object(torch.distributed.rpc.api, '_set_and_start_rpc_agent')
    @dist_init(setup_rpc=False)
    def test_register_rpc_backend_and_set_and_start_rpc_backend(self, mock_rpc_agent, mock_dist_autograd_init):
        backend_name = 'stub_backend'
        backend = rpc.backend_registry.register_backend(backend_name, _stub_construct_rpc_backend_options_handler, _stub_init_rpc_backend_handler)
        with self.assertRaisesRegex(RuntimeError, '^RPC backend .+: already registered$'):
            backend = rpc.backend_registry.register_backend(backend_name, _stub_construct_rpc_backend_options_handler, _stub_init_rpc_backend_handler)
        rpc.init_rpc(name='worker1', backend=backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)

    @dist_init(setup_rpc=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, 'is not unique'):
            store, _, _ = next(torch.distributed.rendezvous(self.init_method, rank=self.rank, world_size=self.world_size))
            rpc._init_rpc_backend(backend=self.rpc_backend, store=store, name='duplicate_name', rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)

    @dist_init(setup_rpc=False)
    def test_duplicate_name_2(self):
        with self.assertRaisesRegex(RuntimeError, 'is not unique'):
            rpc.init_rpc(name=worker_name(self.rank % (self.world_size - 1)), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)

    @dist_init(setup_rpc=False)
    def test_reinit(self):
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        if os.environ.get('RPC_INIT_WITH_TCP', None) == '1' and self.rank == 0:
            expected_reinit_err = 'Address already in use'
        else:
            expected_reinit_err = 'is already initialized'
        with self.assertRaisesRegex(RuntimeError, expected_reinit_err):
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_pg_init_no_rpc_init(self):
        dist.init_process_group(backend='gloo', init_method=self.file_init_method, rank=self.rank, world_size=self.world_size)

        class MyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(3, 4)

            def forward(self, x):
                return self.lin(x)
        model = MyModel()
        model.train()
        model = torch.nn.parallel.DistributedDataParallel(model)
        with self.assertRaisesRegex(RuntimeError, 'Current RPC agent is not set! Did you initialize the RPC framework'):
            params = []
            for param in model.parameters():
                params.append(RRef(param))

    def test_world_size_one(self):
        self._world_size_one(torch.ones(2, 2), torch.ones(2, 2))

    @dist_init(setup_rpc=False)
    def test_invalid_names(self):
        worker_id = 0
        with self.assertRaisesRegex(RuntimeError, 'Worker name must match'):
            info = WorkerInfo('abc*', worker_id)
        with self.assertRaisesRegex(RuntimeError, 'Worker name must match'):
            info = WorkerInfo(' ', worker_id)
        with self.assertRaisesRegex(RuntimeError, 'must be non-empty'):
            info = WorkerInfo('', worker_id)
        with self.assertRaisesRegex(RuntimeError, 'shorter than 128'):
            info = WorkerInfo(''.join(['a' for i in range(500)]), worker_id)

    @dist_init
    def test_worker_info_pickle(self):
        dst_rank = (self.rank + 1) % self.world_size
        worker_info = rpc.api.get_worker_info()
        ret = rpc.rpc_sync(worker_name(dst_rank), identity, args=(worker_info,))
        self.assertEqual(ret, worker_info)

    @dist_init
    def test_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @staticmethod
    def return_callee_id():
        return rpc.get_worker_info().id

    @dist_init
    def test_int_callee(self):
        dst_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(dst_rank, RpcTest.return_callee_id)
        self.assertEqual(ret, dst_rank)

    @dist_init
    def test_add_with_id(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        workder_info = rpc.get_worker_info(worker_name(dst_rank))
        ret = rpc.rpc_sync(workder_info, torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_scalar_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), n))
        self.assertEqual(ret, torch.ones(n, n) + n)

    @dist_init
    def test_async_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_nonzero(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @dist_init
    def test_multi_rpc(self):
        self._multi_rpc(False)

    @dist_init
    def test_future_wait_twice(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        futs = []
        for i in range(20):
            futs.append(rpc.rpc_async(dst, raise_func))
        with self.assertRaisesRegex(ValueError, 'Expected error'):
            torch.futures.wait_all(futs)
        for fut in futs:
            with self.assertRaisesRegex(ValueError, 'Expected error'):
                fut.wait()

    @dist_init(setup_rpc=False)
    def test_wait_all_workers_timeout(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        og_func = rpc.api._wait_all_workers

        def wait_all_workers_sleep(timeout):
            rpc.api._all_gather(SlowPickleClass(0.5), timeout=timeout)
        rpc.api._wait_all_workers = wait_all_workers_sleep
        try:
            with self.assertRaisesRegex(RuntimeError, ''):
                rpc.shutdown(graceful=True, timeout=0.01)
        finally:
            rpc.api._wait_all_workers = og_func
        dist.barrier()

    def test_wait_all_workers_dense(self):
        self._wait_all_workers(heavy_rpc, torch.ones(100, 100))

    def test_wait_all_workers_twice_dense(self):
        self._wait_all_workers_twice(heavy_rpc, torch.ones(100, 100))

    @dist_init
    def test_all_gather(self):
        info = rpc.get_worker_info()
        results = rpc.api._all_gather(info.id)
        expected = {}
        for info in rpc._get_current_rpc_agent().get_worker_infos():
            expected[info.name] = info.id
        self.assertEqual(expected, results)

    @dist_init
    def test_all_gather_timeout(self):
        rpc._set_rpc_timeout(0.1)
        if self.rank == 0:
            with self.assertRaisesRegex(RuntimeError, 'timed out in _all_gather after 0\\.10 seconds'):
                rpc.api._all_gather(SlowPickleClass(0.5))
        else:
            expected_error = self.get_timeout_error_regex()
            with self.assertRaisesRegex(RuntimeError, expected_error):
                rpc.api._all_gather(SlowPickleClass(0.5))

    def _test_barrier_helper(self, info, names, multi_threaded=False):
        names = sorted(names)
        leader = names[0]
        rpc.rpc_sync(leader, _reset_count)
        if not multi_threaded and info.name == leader:
            self.assertEqual(_rpc_barrier_count, 0)
        rpc.api._barrier(names)
        rpc.rpc_sync(leader, _increment_count)
        rpc.api._barrier(names)
        if not multi_threaded and info.name == leader:
            self.assertEqual(_rpc_barrier_count, len(names))

    @dist_init
    def test_rpc_barrier_all(self):
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_subset(self):
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [worker.name for worker in all_worker_info if not worker.id % 2]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_partial_subset(self):
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [f'worker{info.id}']
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_multithreaded(self):
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        threads = []
        for _ in range(3):
            th = threading.Thread(target=self._test_barrier_helper, args=(info, names, True))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

    @dist_init
    def test_graceful_shutdown_with_uneven_workload(self):
        """Test graceful termination."""
        self._run_uneven_workload(heavy_rpc, torch.ones(100, 100))

    @dist_init(setup_rpc=False)
    def test_shutdown_followed_by_rpc(self):
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret, torch.ones(n, n) * 2)
        rpc.shutdown()
        with self.assertRaisesRegex(RuntimeError, '^RPC has not been initialized'):
            rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))

    @dist_init
    def test_expected_src(self):
        dst_rank = (self.rank + 1) % self.world_size
        expected_src_rank = (self.rank - 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), set_value, args=(self.rank,))
        value = VALUE_FUTURE.result()
        self.assertEqual(value, expected_src_rank)

    @dist_init
    def test_py_built_in(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), my_function, kwargs={'a': n, 'b': n + 1, 'c': n + 2})
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    def test_build_rpc_profiling_key(self):
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            rpc_profiling_key = _build_rpc_profiling_key(exec_mode, 'foo', 'worker0', 'worker1')
            self.assertIn(exec_mode.value, rpc_profiling_key)
            self.assertIn('foo', rpc_profiling_key)
            self.assertIn('worker0', rpc_profiling_key)
            self.assertIn('worker1', rpc_profiling_key)

    def check_profiling_info(self, self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode):
        self.assertTrue(self_worker_name in rpc_event.name)
        self.assertTrue(dst_worker_name in rpc_event.name)
        if isinstance(func, torch.jit.ScriptFunction):
            self.assertTrue(torch._jit_internal._qualified_name(func) in rpc_event.name)
        else:
            self.assertTrue(func.__name__ in rpc_event.name)
        self.assertTrue(rpc_exec_mode.value in rpc_event.name)
        self.assertEqual(rpc_event.count, 1)

    @dist_init
    def test_profiler_rpc_record_shapes(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        t1, t2 = (torch.ones(100), torch.ones(100))
        with _profile(record_shapes=True) as prof:
            rpc.rpc_sync(dst_worker, torch.add, args=(t1, t2))
        function_events = prof.function_events
        remote_events = [event for event in function_events if event.is_remote]
        remote_add_event = next((event for event in remote_events if 'aten::add' in event.name))
        remote_add_input_shapes = remote_add_event.input_shapes
        with _profile(record_shapes=True) as prof:
            torch.add(t1, t2)
        local_function_events = prof.function_events
        local_add_event = next((event for event in local_function_events if 'aten::add' in event.name))
        local_add_input_shapes = local_add_event.input_shapes
        self.assertEqual(remote_add_input_shapes, local_add_input_shapes)

    @dist_init
    def test_profiler_rpc_memory(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with _profile(profile_memory=True) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            res = fut.wait()
        function_events = p.function_events
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        self.assertNotEqual({0}, event_cpu_mem_usages)
        with _profile(profile_memory=False) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            res = fut.wait()
        function_events = p.function_events
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        self.assertEqual({0}, event_cpu_mem_usages)

    @dist_init
    def test_profiler_export_trace(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with _profile() as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            res = fut.wait()
        events = p.function_events
        with TemporaryFileName() as fname:
            path = fname
            p.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
                event_names = [event['name'] for event in trace]
                for expected_event_name in EXPECTED_REMOTE_EVENTS + [RPCExecMode.ASYNC.value]:
                    event_exists = any((expected_event_name in event_name for event_name in event_names))
                    self.assertTrue(event_exists)

    @dist_init
    def test_profiler_rpc_key_names(self):
        if self.rank != 1:
            return
        dst_ranks = [rank for rank in range(0, self.world_size) if rank != self.rank]

        def rpc_with_profiling(dst_worker):
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                fut.wait()
            events = prof.function_events
            remote_event_names = {event.name: event for event in events if event.is_remote}
            rpc_profiling_key = _build_rpc_profiling_key(RPCExecMode.ASYNC, udf_with_torch_ops.__qualname__, worker_name(self.rank), dst_worker)
            remote_event_name_set = set(EXPECTED_REMOTE_EVENTS)
            for name, event in remote_event_names.items():
                self.assertTrue(name.startswith(rpc_profiling_key))
                self.assertTrue(event.is_remote)
                self.assertTrue(event.node_id == rpc.get_worker_info(dst_worker).id)
                operator_name_substr = name[len(rpc_profiling_key):]
                matching_event = {remote_event_name for remote_event_name in remote_event_name_set if remote_event_name in operator_name_substr}
                remote_event_name_set -= matching_event
            self.assertTrue(remote_event_name_set == set(), f'Expected {remote_event_name_set} to be included in remote profiler output.')
        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            num_parallel_rpcs = 2
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_rpcs) as executor:
                futs = [executor.submit(rpc_with_profiling, dst_worker) for _ in range(num_parallel_rpcs)]
                for fut in futs:
                    fut.result()

    def _run_test_profiler_remote_events_profiled(self):
        if self.rank != 1:
            return
        dst_ranks = [rank for rank in range(0, self.world_size) if rank != self.rank]
        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                ret = fut.wait()
            events = prof.function_events
            rpc_event = get_function_event(events, RPCExecMode.ASYNC.value)
            self.check_profiling_info(worker_name(self.rank), dst_worker, udf_with_torch_ops, rpc_event, RPCExecMode.ASYNC)
            remote_events = {event.name: event for event in events if event.is_remote}
            rpc_profiling_key = _build_rpc_profiling_key(RPCExecMode.ASYNC, udf_with_torch_ops.__qualname__, worker_name(self.rank), worker_name(dst))
            for expected_remote_event_name in EXPECTED_REMOTE_EVENTS:
                expected_key = rpc_profiling_key + REMOTE_OP_STR + expected_remote_event_name
                self.assertTrue(expected_key in remote_events)
                remote_event = remote_events[expected_key]
                self.assertEqual(remote_event.node_id, dst)

            def convert_remote_to_local(event_name):
                remote_op_key = rpc_profiling_key + REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key):]
            remote_events_list = [convert_remote_to_local(event.name) for event in events if convert_remote_to_local(event.name) in EXPECTED_REMOTE_EVENTS]
            self.assertEqual(set(remote_events_list), set(EXPECTED_REMOTE_EVENTS), f'Mismatch between profiled events: {set(remote_events_list)} and expected events: {set(EXPECTED_REMOTE_EVENTS)}')

    @dist_init
    def test_profiler_remote_events_profiled(self):
        self._run_test_profiler_remote_events_profiled()

    @dist_init
    def test_profiler_remote_events_profiled_single_threaded(self):
        self._run_test_profiler_remote_events_profiled()

    def run_profiling_workload(self, dst):
        fut = rpc.rpc_async(worker_name(dst), torch.mul, args=(torch.tensor(1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)))
        fut.wait()

    def _run_rpc_profiling_async_function(self, device='cpu'):
        if self.rank != 1:
            return
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        x = torch.ones(2)
        y = torch.ones(2)
        with _profile() as prof:
            ret = rpc.rpc_async(dst1, slow_async_add, args=(dst2, x, y, device), timeout=20)
            out = ret.wait()
        function_events = prof.function_events
        key_prefix = _build_rpc_profiling_key(RPCExecMode.ASYNC, slow_async_add.__qualname__, worker_name(self.rank), dst1)
        nested_rpc_key_prefix = _build_rpc_profiling_key(RPCExecMode.ASYNC, slow_add.__qualname__, dst1, dst2)
        expected_key = key_prefix + REMOTE_OP_STR + nested_rpc_key_prefix
        remote_events = [event for event in function_events if event.is_remote]
        rpc_remote_event = [event for event in remote_events if event.name == expected_key]
        self.assertEqual(1, len(rpc_remote_event))
        rpc_remote_event = rpc_remote_event[0]
        self.assertEqual(rpc_remote_event.node_id, (self.rank + 1) % self.world_size)
        remote_add_key = expected_key + REMOTE_OP_STR + torch.jit._builtins._find_builtin(torch.add)
        remote_add_event = [event for event in remote_events if event.name == remote_add_key]
        self.assertEqual(1, len(remote_add_event))
        remote_add_event = remote_add_event[0]
        self.assertEqual(remote_add_event.node_id, (self.rank + 2) % self.world_size)

    @dist_init
    def test_rpc_profiling_async_function(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device='cuda:0')

    @dist_init
    def test_rpc_profiling_async_function_single_threaded(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device='cuda:0')

    @dist_init
    def test_rpc_profiling_remote_record_function(self):
        if self.rank != 1:
            return
        dst_ranks = [i for i in range(self.world_size) if i != self.rank]
        for dst_rank in dst_ranks:
            dst_worker = worker_name(dst_rank)
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=(-1, True))
                fut.wait()
            function_events = prof.function_events
            record_function_remote_event = [evt for evt in function_events if '##forward##' in evt.name]
            self.assertEqual(1, len(record_function_remote_event))
            record_function_remote_event = record_function_remote_event[0]
            self.assertEqual(record_function_remote_event.node_id, dst_rank)

            def get_cpu_children(event):
                if not event.cpu_children:
                    return []
                cpu_children = event.cpu_children
                for e in event.cpu_children:
                    cpu_children.extend(get_cpu_children(e))
                return cpu_children
            remote_children = get_cpu_children(record_function_remote_event)
            with _profile() as prof:
                udf_with_torch_ops(-1, True)
            local_function_events = prof.function_events
            local_record_function_event = next((evt for evt in local_function_events if '##forward##' in evt.name))
            local_children = get_cpu_children(local_record_function_event)
            local_children_names = [evt.name for evt in local_children]
            REMOTE_OP_STR = '#remote_op: '

            def convert_remote_to_local(event_name):
                remote_op_key = REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key):]
            for evt in remote_children:
                local_name = convert_remote_to_local(evt.name)
                self.assertTrue(local_name in local_children_names)

    def validate_profiling_workload(self, dst, prof):

        def convert_remote_to_local(event_name):
            return event_name[event_name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR):]
        events = prof.function_events
        remote_events = {convert_remote_to_local(event.name): event for event in events if event.is_remote}
        self.assertTrue('aten::mul' in remote_events)
        remote_mul_event = remote_events['aten::mul']
        self.assertEqual(remote_mul_event.node_id, dst)
        self.check_profiling_info(worker_name(self.rank), worker_name(dst), torch.mul, remote_mul_event, RPCExecMode.ASYNC)

    def _run_test_profiler_with_autograd_context(self):
        dst = (self.rank + 1) % self.world_size
        if self.rank == 1:
            with dist_autograd.context() as context_id:
                with _profile() as prof:
                    self.run_profiling_workload(dst)
            self.validate_profiling_workload(dst, prof)
            with _profile() as prof:
                with dist_autograd.context() as context_id:
                    self.run_profiling_workload(dst)
            self.validate_profiling_workload(dst, prof)

    @dist_init
    def test_profiler_with_autograd_context_single_threaded(self):
        self._run_test_profiler_with_autograd_context()

    @dist_init
    def test_profiler_with_autograd_context(self):
        self._run_test_profiler_with_autograd_context()

    def _profiler_test_with_rpc(self, rpc_exec_mode, func, args, use_record_function=False, dst=None, kineto_profile=False):
        dst = dst if dst is not None else (self.rank + 1) % self.world_size
        p = _profile if not kineto_profile else torch.profiler.profile
        if self.rank == 1:
            with p() as prof:
                record_function_ctx_mgr = contextlib.nullcontext() if not use_record_function else torch.autograd.profiler.record_function('foo')
                with record_function_ctx_mgr as rf:
                    if rpc_exec_mode == RPCExecMode.SYNC:
                        rpc.rpc_sync(worker_name(dst), func, args=args)
                    elif rpc_exec_mode == RPCExecMode.ASYNC:
                        fut = rpc.rpc_async(worker_name(dst), func, args=args)
                        if kineto_profile:
                            fut2 = rpc.rpc_async(worker_name(dst), func, args=args)
                            fut2.wait()
                        fut.wait()
                    else:
                        self.assertTrue(rpc_exec_mode == RPCExecMode.REMOTE)
                        rref = rpc.remote(worker_name(dst), func, args=args)
                        rref.to_here()
                        rref._get_profiling_future().wait()
            events = prof.function_events if not kineto_profile else prof.events()
            if kineto_profile:
                with self.assertRaises(IndexError):
                    get_function_event(events, rpc_exec_mode.value)
                return
            rpc_event = get_function_event(events, rpc_exec_mode.value)
            self.assertEqual(rpc_event.node_id, self.rank)
            remote_events = {event for event in events if event.node_id == dst} - {rpc_event}
            self.assertGreaterEqual(len(remote_events), 1)
            for remote_event in remote_events:
                self.assertEqual(remote_event.node_id, dst)
            if use_record_function:
                scope_event = get_function_event(events, 'foo')
                self.assertLessEqual(scope_event.time_range.start, rpc_event.time_range.start)
                self.assertGreaterEqual(scope_event.time_range.end, rpc_event.time_range.end)
            self_worker_name = worker_name(self.rank)
            dst_worker_name = worker_name(dst)
            self.check_profiling_info(self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode)
            if use_record_function:
                foo_event_ix = next((i for i, event in enumerate(events) if 'foo' in event.name))
                rpc_event_idx = next((i for i, event in enumerate(events) if rpc_exec_mode.value in event.name))
                self.assertLess(foo_event_ix, rpc_event_idx)

    def _run_test_profiler_with_sync_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,), use_record_function=True)

    @dist_init
    def test_profiler_with_sync_rpc_udf(self):
        self._run_test_profiler_with_sync_rpc_udf()

    @dist_init
    def test_profiler_with_sync_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_udf()

    def _run_test_profiler_with_sync_rpc_builtin(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1)))
        self._profiler_test_with_rpc(RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1)), use_record_function=True)

    @dist_init
    def test_profiler_with_sync_rpc_builtin(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    @dist_init
    def test_profiler_with_sync_rpc_builtin_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    def _run_test_profiler_with_async_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,), use_record_function=True)
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,), kineto_profile=True)

    @dist_init
    def test_profiler_with_async_rpc_udf(self):
        self._run_test_profiler_with_async_rpc_udf()

    @dist_init
    def test_profiler_with_async_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_async_rpc_udf()

    def _run_test_profiler_with_async_rpc_builtin(self):
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1)))
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1)), use_record_function=True)

    @dist_init
    def test_profiler_with_async_rpc_builtin(self):
        self._run_test_profiler_with_async_rpc_builtin()

    @dist_init
    def test_profiler_with_async_rpc_builtin_single_threaded(self):
        self._run_test_profiler_with_async_rpc_builtin()

    def _run_test_profiler_with_remote_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,), use_record_function=True)
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,), dst=self.rank)

    @dist_init
    def test_profiler_with_remote_udf(self):
        self._run_test_profiler_with_remote_udf()

    @dist_init
    def test_profiler_with_remote_udf_single_threaded(self):
        self._run_test_profiler_with_remote_udf()

    def _run_test_profiler_with_remote_builtin(self):
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1)))
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1)), use_record_function=True)
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1)), dst=self.rank)

    @dist_init
    def test_profiler_with_remote_builtin(self):
        self._run_test_profiler_with_remote_builtin()

    @dist_init
    def test_profiler_with_remote_builtin_single_threaded(self):
        self._run_test_profiler_with_remote_builtin()

    def _run_test_profiler_with_script_async_rpc(self):
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_script_func, args=(torch.tensor(1),))
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_script_func, args=(torch.tensor(1),), use_record_function=True)

    @dist_init
    def test_profiler_with_script_async_rpc(self):
        self._run_test_profiler_with_script_async_rpc()

    @dist_init
    def test_profiler_with_script_async_rpc_single_threaded(self):
        self._run_test_profiler_with_script_async_rpc()

    def _run_test_profiler_with_script_sync_rpc(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_script_func, args=(torch.tensor(1),))
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_script_func, args=(torch.tensor(1),), use_record_function=True)

    @dist_init
    def test_profiler_with_script_sync_rpc(self):
        self._run_test_profiler_with_script_sync_rpc()

    @dist_init
    def test_profiler_with_script_sync_rpc_single_threaded(self):
        self._run_test_profiler_with_script_sync_rpc()

    def _run_test_profiler_with_script_remote_rpc(self):
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),))
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),), use_record_function=True)
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),), dst=self.rank)

    @dist_init
    def test_profiler_with_script_remote_rpc(self):
        self._run_test_profiler_with_script_remote_rpc()

    @dist_init
    def test_profiler_with_script_remote_rpc_single_threaded(self):
        self._run_test_profiler_with_script_remote_rpc()

    def _assert_top_level_events(self, process_global_events, expected_top_level_event_names):
        top_level_event_names = []
        for thread_local_events in process_global_events:
            last_end_time = 0
            for event in thread_local_events:
                event_name = event.name
                time_range = event.time_range
                if time_range.start > last_end_time:
                    top_level_event_names.append(event_name)
                    last_end_time = time_range.end
        top_level_event_names = sorted(top_level_event_names)
        expected_top_level_event_names = sorted(expected_top_level_event_names)
        self.assertEqual(top_level_event_names, expected_top_level_event_names, f'Expected events {expected_top_level_event_names}, but got {top_level_event_names}')

    @dist_init
    def test_server_process_global_profiler(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)
        x = torch.tensor(1)
        y = torch.tensor(2)
        outer_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
        outer_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        inner_profile_rref = rpc.remote(dst_worker_name, rpc._server_process_global_profile)
        inner_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        inner_profile_rref.rpc_sync().__exit__(None, None, None)
        outer_profile_rref.rpc_sync().__exit__(None, None, None)
        inner_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (inner_profile_rref,))
        expected_inner_events = ['aten::sub']
        expected_outer_events = expected_inner_events + ['aten::add']
        self._assert_top_level_events(inner_events, expected_inner_events)
        outer_events = rpc.rpc_sync(dst_worker_name, get_events_from_profile, (outer_profile_rref,))
        self._assert_top_level_events(outer_events, expected_outer_events)
        inner_profile_rref.rpc_sync().key_averages()
        outer_profile_rref.rpc_sync().key_averages()

    @dist_init
    def test_async_record_function_double_end_callbacks(self):
        num_sleep_seconds = 1
        if self.rank == 1:
            with _profile() as pf:
                with torch.autograd.profiler.record_function('foo') as rf:
                    fut = rpc.rpc_async(worker_name(0), my_sleep_func, args=(num_sleep_seconds,))
                    rf._call_end_callbacks_on_future(fut)
                    with self.assertRaisesRegex(RuntimeError, 'can only be called once.'):
                        rf._call_end_callbacks_on_future(fut)
                fut.wait()

    @dist_init
    def test_async_record_function_legacy(self):
        num_sleep_seconds = 1
        if self.rank == 1:
            with _profile() as pf:
                try:
                    handle = torch.ops.profiler._record_function_enter('foo', None)
                    fut = rpc.rpc_async(worker_name(0), my_sleep_func, args=(num_sleep_seconds,))
                    torch.ops.profiler._call_end_callbacks_on_jit_fut(handle, fut)
                finally:
                    torch.ops.profiler._record_function_exit(handle)
                fut.wait()

    @dist_init
    def test_async_record_function_cbs_jit_call(self):
        if self.rank == 1:
            with _profile() as pf:
                key = _build_rpc_profiling_key(RPCExecMode.ASYNC, torch._jit_internal._qualified_name(my_script_func), 'worker1', 'worker0')
                with torch.autograd.profiler.record_function(key) as rf:
                    fut = rpc.rpc_async(worker_name(0), my_script_func, args=(torch.tensor(1),))
                    fut = torch.ops.profiler._call_end_callbacks_on_jit_fut(rf.record, fut)
                result = fut.wait()
                expected = torch.add(torch.tensor(1), torch.tensor(1))
                self.assertEqual(result, expected)
            events = pf.function_events
            rpc_event = get_function_event(events, torch._jit_internal._qualified_name(my_script_func))
            self.assertTrue(torch._jit_internal._qualified_name(my_script_func) in rpc_event.name)

    @dist_init
    def test_py_class_constructor(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass, args=(n,))
        self.assertEqual(ret.a, n)

    @dist_init
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass(2).my_instance_method, args=(n,))
        self.assertEqual(ret, MyClass(2).my_instance_method(n))

    @dist_init
    def test_py_class_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass.my_class_method, args=(n, n + 1))
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))

    @dist_init
    def test_py_class_static_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass.my_static_method, args=(n + 10,))
        self.assertEqual(ret, MyClass.my_static_method(n + 10))

    @dist_init
    def test_py_multi_async_call(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker_info = rpc.get_worker_info(worker_name(dst_rank))
        fut1 = rpc.rpc_async(dst_worker_info, MyClass.my_static_method, args=(n + 10,))
        fut2 = rpc.rpc_async(dst_worker_info, min, args=(n, n + 1, n + 2))
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @dist_init
    def test_py_no_return_result(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), no_result)
        self.assertEqual(ret, no_result())

    @dist_init
    def test_py_tensors(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), my_tensor_function, args=(torch.ones(n, n), torch.ones(n, n)))
        self.assertEqual(ret, my_tensor_function(torch.ones(n, n), torch.ones(n, n)))

    @dist_init
    def test_py_tensors_multi_async_call(self):
        futs = []
        n = self.rank + 1
        dst_rank = n % self.world_size
        for i in range(100):
            fut = rpc.rpc_async(worker_name(dst_rank), my_tensor_function, args=(torch.ones(i, i), torch.ones(i, i)))
            futs.append(fut)
        j = 0
        for val in torch.futures.wait_all(futs):
            self.assertEqual(val, my_tensor_function(torch.ones(j, j), torch.ones(j, j)))
            j += 1

    @dist_init
    def test_py_tensors_in_container(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [torch.ones(n, n), torch.ones(n, n)]
        b = TensorClass(build_complex_tensors())
        c = {'foo': torch.ones(n, n), 'bar': torch.ones(n, n)}
        ret = rpc.rpc_sync(worker_name(dst_rank), my_complex_tensor_function, args=(a, b, c))
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    @dist_init
    def test_py_nested_pickle(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), run_nested_pickle, args=(MyPickleClass(), torch.ones(2, 2)))
        m = MyPickleClass()
        m.set(my_tensor_function(torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(ret, run_nested_pickle(m, torch.ones(2, 2)))

    @dist_init
    def test_py_function_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        with self.assertRaises(TypeError):
            ret = rpc.rpc_sync(worker_name(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        with captured_output() as (_, err):
            initialize_pg(self.file_init_method, self.rank, self.world_size)
            dist.barrier()
            n = self.rank + 1
            dst_rank = n % self.world_size
            fut = rpc.rpc_async(worker_name(dst_rank), raise_func)
            with self.assertRaisesRegex(ValueError, expected_err):
                fut.wait()
            dist.barrier()
        stderr_lines = err.getvalue()
        self.assertTrue(expected_err in stderr_lines)

    @dist_init
    def test_py_raise_in_user_func_escaped_str(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(worker_name(dst_rank), raise_func_escape)
        try:
            fut.wait()
        except ValueError as e:
            msg = str(e)
            self.assertEqual(msg, msg.encode('utf-8').decode('unicode_escape'))
        else:
            self.assertTrue(False, 'expected raise_func_escape to raise ValueError.')

    @dist_init
    def test_nested_rpc(self):
        self._nested_rpc(nested_rpc, torch.ones(2, 2) + 1)

    @dist_init
    def test_stress_light_rpc(self):
        self._stress_test_rpc(light_rpc)

    @dist_init
    def test_stress_heavy_rpc(self):
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_stress_heavy_rpc_torchscript(self):
        self._stress_test_rpc(heavy_rpc_torchscript, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_builtin_remote_ret(self):
        self._builtin_remote_ret(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2)

    @dist_init
    def test_builtin_remote_self(self):
        self._builtin_remote_self(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2)

    @staticmethod
    def _multi_args_fn(n, sparse=False):
        if sparse:
            return (build_sparse_tensor(), build_sparse_tensor())
        else:
            return (torch.ones(n, n), torch.ones(n, n))

    @dist_init
    def test_multi_builtin_remote_ret(self):
        self._test_multi_remote_call(torch.add, False, args_fn=RpcTest._multi_args_fn)

    @dist_init
    def test_py_udf_remote(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(worker_name(dst_rank), my_function, kwargs={'a': n, 'b': n + 1, 'c': n + 2})
        self.assertEqual(rref.to_here(), my_function(n, n + 1, n + 2))

    @staticmethod
    def _multi_kwargs_fn(n, sparse=False):
        if sparse:
            return {'a': build_sparse_tensor(), 'b': build_sparse_tensor(), 'c': build_sparse_tensor()}
        else:
            return {'a': torch.ones(n, n), 'b': torch.ones(n, n), 'c': torch.ones(n, n)}

    @dist_init
    def test_multi_py_udf_remote(self):
        self._test_multi_remote_call(my_function, False, kwargs_fn=RpcTest._multi_kwargs_fn)

    @dist_init
    def test_py_rref_args(self):
        self._py_rref_args(torch.ones(2, 2), 1, torch.ones(2, 2), 2, torch.ones(2, 2) * 2 + 3)

    @dist_init
    def test_py_rref_args_user_share(self):
        self._py_rref_args_user_share(torch.ones(2, 2), 1, 2, torch.ones(2, 2), 3, 4, torch.ones(2, 2) * 2 + 10)

    @dist_init
    def test_py_rpc_rref_args(self):
        self._py_rpc_rref_args(torch.ones(2, 2), 1, 2, torch.ones(2, 2), 3, 4, torch.ones(2, 2) * 2 + 10)

    @dist_init
    def test_nested_remote(self):
        self._nested_remote(nested_remote, torch.ones(2, 2) + 3)

    @dist_init
    def test_nested_rref(self):
        self._nested_rref(nested_rref, torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)

    @dist_init
    def test_nested_rref_stress(self):
        self._nested_rref_stress(nested_rref, torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)

    @dist_init
    def test_multi_layer_nested_async_rpc(self):
        ttl = 20
        n = self.rank + 1
        dst_rank = n % self.world_size
        multi_layer_nested_async_rpc(dst_rank, self.world_size, ttl)

    @dist_init
    def test_remote_with_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(worker_name(dst_rank), raise_func)
        with self.assertRaises(ValueError):
            rref.to_here()
        rref = rpc.remote(worker_name(self.rank), no_result, args=(10,))
        with self.assertRaises(TypeError):
            rref.to_here()

    @dist_init
    def test_rpc_return_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref = rpc.rpc_sync(worker_name(dst_rank1), rpc_return_rref, args=(worker_name(dst_rank2),))
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_rref_forward_chain(self):
        ttl = 8
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1))
        ret_rref = rref_forward_chain(dst_rank, self.world_size, rref, ttl)
        for i in range(ttl):
            self.assertEqual(len(ret_rref), 1)
            ret_rref = ret_rref[0].to_here()
        ret = ret_rref
        self.assertEqual(ret, torch.add(torch.ones(n, n), 1))

    @dist_init
    def test_local_rref_no_fork(self):
        local_rref = RRef(35)
        self.assertEqual(local_rref.local_value(), 35)

    @dist_init
    def test_local_value_not_on_owner(self):
        next_rank = (self.rank + 1) % self.world_size
        rref = rpc.remote(worker_name(next_rank), torch.add, args=(torch.ones(1), torch.ones(1)))
        with self.assertRaisesRegex(RuntimeError, f"For UserRRef\\(rref_id=GloballyUniqueId\\(created_on={self.rank}, local_id=0\\), fork_id=GloballyUniqueId\\(created_on={self.rank}, local_id=1\\)\\), can't call localValue\\(\\) on user WorkerInfo\\(id={self.rank}, name={worker_name(self.rank)}\\). Call it on owner WorkerInfo\\(id={next_rank}, name={worker_name(next_rank)}\\)"):
            rref.local_value()

    @dist_init
    def test_return_local_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_list = rpc.rpc_sync(worker_name(dst_rank), get_rref_list, args=([1, 2, 3],))
        for rref in rref_list:
            rpc.rpc_sync(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, 10))
        rets = [rpc.rpc_sync(rref.owner(), _call_method_on_rref, args=(MyClass.get_value, rref)) for rref in rref_list]
        self.assertEqual(rets, [11, 12, 13])

    @dist_init
    def _test_rref_type(self, blocking):

        def launched_rpc(events):
            expected_name = f'rpc_{RPCExecMode.ASYNC.value}#_rref_typeof_on_owner'
            return any((e.name.startswith(expected_name) for e in events))
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, torch.add, args=(torch.ones(2), 1))
        with _profile() as p:
            t = rref._get_type(blocking=blocking)
            if not blocking:
                t = t.wait()
        self.assertTrue(launched_rpc(p.function_events))
        expected_type = type(torch.ones(2))
        self.assertEqual(t, expected_type)
        futs = []

        def verify(fut):
            self.assertEqual(fut.value(), expected_type)
        with _profile() as p:
            for _ in range(10):
                t = rref._get_type(blocking=blocking)
                if not blocking:
                    futs.append(t)
                    t.add_done_callback(verify)
                    t = t.wait()
                self.assertEqual(t, expected_type)
        if not blocking:
            first_fut = futs[0]
            for f in futs[1:]:
                self.assertTrue(f is first_fut)
        self.assertFalse(launched_rpc(p.function_events))
        self.assertEqual(t, type(torch.ones(2)))
        rref = rpc.remote(dst, MyClass, args=(0,))
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_blocking(self):
        self._test_rref_type(blocking=True)

    def test_rref_type_non_blocking(self):
        self._test_rref_type(blocking=False)

    @dist_init
    def _test_rref_type_with_error(self, blocking):
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, raise_func)
        if blocking:
            with self.assertRaisesRegex(ValueError, 'Expected error'):
                rref._get_type(blocking=blocking)
        else:
            fut = rref._get_type(blocking=blocking)
            with self.assertRaisesRegex(ValueError, 'Expected error'):
                fut.wait()

    def test_rref_type_with_error_blocking(self):
        self._test_rref_type_with_error(blocking=True)

    def test_rref_type_with_error_non_blocking(self):
        self._test_rref_type_with_error(blocking=False)

    @dist_init
    def _test_rref_type_owner(self, blocking):
        rref = RRef(torch.ones(2) + 1)
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, type(torch.ones(2)))
        rref = RRef(MyClass(0))
        rref_type = rref._get_type(blocking=blocking)
        if not blocking:
            rref_type = rref_type.wait()
        self.assertEqual(rref_type, MyClass)

    def test_rref_type_owner_blocking(self):
        self._test_rref_type_owner(blocking=True)

    def test_rref_type_owner_non_blocking(self):
        self._test_rref_type_owner(blocking=False)

    @staticmethod
    def _slow_add(x, y):
        time.sleep(1)
        return x + y

    @dist_init
    def test_rref_type_slow_init(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, RpcTest._slow_add, args=(torch.ones(2), 1))
        self.assertEqual(rref._get_type(), type(torch.ones(2)))

    @dist_init
    def test_owner_equality(self):
        a = RRef(40)
        b = RRef(50)
        other_rank = (self.rank + 1) % self.world_size
        other_a = rpc.remote(worker_name(other_rank), torch.add, args=(torch.ones(1), 1))
        other_b = rpc.remote(worker_name(other_rank), torch.add, args=(torch.ones(1), 1))
        other_a.to_here()
        other_b.to_here()
        self.assertNotEqual(a.owner(), 23)
        self.assertEqual(other_a.owner(), other_b.owner())
        self.assertNotEqual(a.owner(), other_a.owner())
        self.assertEqual(other_a.owner(), other_a.owner())
        self.assertEqual(other_a.owner(), other_b.owner())
        self.assertEqual(a.owner(), a.owner())
        self.assertEqual(a.owner(), b.owner())
        self.assertEqual(a.owner(), rpc.get_worker_info())
        x = {}
        x[a.owner()] = a
        x[other_a.owner()] = other_a
        self.assertEqual(x[a.owner()], a)
        self.assertEqual(x[b.owner()], a)
        self.assertEqual(x[other_a.owner()], other_a)
        self.assertEqual(x[other_b.owner()], other_a)
        self.assertEqual(len(x), 2)

    @dist_init
    def test_pass_local_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker = worker_name(dst_rank)
        rref = RRef(40)
        self.assertEqual(rpc.rpc_sync(dst_worker, add_rref_to_value, args=(rref, 50)), 90)
        self.assertEqual(rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 50)).wait(), 90)
        self.assertEqual(rpc.remote(dst_worker, add_rref_to_value, args=(rref, 50)).to_here(), 90)

    @dist_init
    def test_remote_same_worker(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 2))
        rref_b = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1))
        rref_c = rpc.remote(worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b))
        self.assertEqual(rref_c.to_here(), torch.ones(n, n) + 4)

    @dist_init(setup_rpc=True)
    def test_call_method_on_rref(self):
        """
        Tests that it is possible to call an instance method on a remote object
        by using rref.owner() as destination of the call.
        """
        vals = [10, 2, 5, 7]
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst_rank)
        rref = rpc.remote(dst_worker, MyClass, args=(vals[0],))
        rpc.rpc_sync(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, vals[1]))
        rpc.rpc_async(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, vals[2])).wait()
        rpc.remote(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, vals[3])).to_here()
        result = rpc.rpc_sync(dst_worker, _call_method_on_rref, args=(MyClass.get_value, rref))
        self.assertEqual(result, sum(vals))

    @mock.patch.object(torch.distributed.rpc.api, '_delete_all_user_and_unforked_owner_rrefs')
    def _test_rref_leak(self, _mock_delete_all_user_and_unforked_owner_rrefs, ignore_leak):
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), torch.add, args=(torch.ones(2, 2), 1))
        import torch.distributed.rpc.api as api
        if ignore_leak:
            api._ignore_rref_leak = True
            rpc.shutdown(graceful=True)
        else:
            api._ignore_rref_leak = False
            with self.assertRaisesRegex(RuntimeError, 'Leaking RRef'):
                rpc.shutdown(graceful=True)

    @dist_init(setup_rpc=False)
    def test_rref_leak(self):
        self._test_rref_leak(ignore_leak=False)

    @dist_init(setup_rpc=False)
    def test_ignore_rref_leak(self):
        self._test_rref_leak(ignore_leak=True)

    @dist_init
    def test_rref_str(self):
        rref1 = RRef(self.rank)
        id_class = 'GloballyUniqueId'
        self.assertEqual(f'OwnerRRef({id_class}(created_on={self.rank}, local_id=0))', rref1.__str__())
        dst_rank = (self.rank + 1) % self.world_size
        rref2 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(rref2.__str__(), 'UserRRef(RRefId = {0}(created_on={1}, local_id=1), ForkId = {0}(created_on={1}, local_id=2))'.format(id_class, self.rank))

    @dist_init
    def test_rref_get_future(self):
        if self.rank == 0:
            rref = rpc.remote(worker_name(1), torch.add, args=(1, 1))
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)
            rref = rpc.remote(worker_name(1), foo_add, args=())
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)
            rref = rpc.remote(worker_name(1), my_script_func, args=(torch.tensor(1),))
            rref.to_here()
            fut = rref._get_future()
            self.assertIsInstance(fut, torch._C.Future)

    @dist_init
    def test_rref_context_debug_info(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rref1 = RRef(self.rank)
        info = _rref_context_get_debug_info()
        self.assertIn('num_owner_rrefs', info)
        self.assertIn('num_pending_users', info)
        self.assertEqual(0, int(info['num_owner_rrefs']))
        self.assertEqual(0, int(info['num_pending_users']))
        dist.barrier()
        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), set_global_rref, args=(rref1,))
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()
        info = _rref_context_get_debug_info()
        self.assertIn('num_owner_rrefs', info)
        self.assertEqual(1, int(info['num_owner_rrefs']))
        self.assertEqual(0, int(info['num_pending_users']))
        dist.barrier()
        rpc.rpc_sync(worker_name(dst_rank), clear_global_rref)
        while int(info['num_owner_rrefs']) != 0:
            info = _rref_context_get_debug_info()
            time.sleep(0.1)
        dist.barrier()
        rref2 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
        rref3 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
        rref2.to_here()
        rref3.to_here()
        wait_until_pending_futures_and_users_flushed()
        dist.barrier()
        info = _rref_context_get_debug_info()
        self.assertIn('num_owner_rrefs', info)
        self.assertEqual(2, int(info['num_owner_rrefs']))
        self.assertEqual(0, int(info['num_pending_users']))
        dist.barrier()

    @dist_init
    def test_disable_gil_profiling(self):
        dst_rank = (self.rank + 1) % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1)))
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertRaises(KeyError, lambda: info['agent.gil_average_wait_time_us'])
        rpc.enable_gil_profiling(True)
        rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(1), torch.ones(1)))
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertIn('agent.gil_average_wait_time_us', info)

    @dist_init(setup_rpc=False)
    def test_local_shutdown(self):
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        rpc.shutdown(graceful=False)

    @dist_init
    def test_debug_info(self):
        import torch.distributed.autograd as dist_autograd
        info = _get_debug_info()
        rref_info = _rref_context_get_debug_info()
        agent_info = rpc.api._get_current_rpc_agent().get_debug_info()
        autograd_info = dist_autograd._get_debug_info()
        common_keys = rref_info.keys() & agent_info.keys() & autograd_info.keys()
        self.assertEqual(0, len(common_keys))
        expected = {}
        expected.update(rref_info)
        expected.update(agent_info)
        expected.update(autograd_info)
        for key in expected.keys():
            self.assertIn(key, info.keys())
        for key in info.keys():
            self.assertIn(key, expected.keys())

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(IS_MACOS, 'Test is flaky on MacOS since libuv error handling is not as robust as TCP')
    def test_handle_send_exceptions(self):
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        rpc._set_rpc_timeout(10)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        if self.rank == 1:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker = worker_name(dst_rank)
            error_str = self.get_shutdown_error_regex()
            wait_until_node_failure(dst_rank, error_str)
            fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.ones(1), 3))
            with self.assertRaisesRegex(RuntimeError, error_str):
                fut.wait()
        rpc.shutdown(graceful=False)

    @dist_init
    def test_deadlock(self):
        if self.rank == 1:
            dst1 = worker_name((self.rank + 1) % self.world_size)
            x = torch.ones(2)
            y = torch.ones(2)
            rpc.rpc_async(dst1, RpcTest._slow_add, args=(x, y), timeout=15).wait()
        dist_initialized = dist.is_initialized()
        if not dist_initialized:
            dist.init_process_group(backend='gloo', init_method=self.file_init_method, rank=self.rank, world_size=self.world_size)

    @dist_init(setup_rpc=False)
    def test_local_shutdown_with_rpc(self):
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        n = self.rank + 1
        dst_rank = n % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        rpc.shutdown(graceful=False)

    @dist_init(setup_rpc=False)
    def test_set_and_get_default_rpc_timeout(self):
        timeout = 0.5
        rpc_backend_options = self.rpc_backend_options
        rpc_backend_options.rpc_timeout = timeout
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)
        set_timeout = rpc.get_rpc_timeout()
        self.assertEqual(timeout, set_timeout)
        rpc.shutdown()

    @dist_init
    def test_default_timeout_used(self):
        """
        Tests that if no timeout is passed into rpc_async and rpc_sync, then the
        default timeout is used.
        """
        dst_rank = (self.rank + 1) % self.world_size
        rpc._set_rpc_timeout(0.001)
        futs = [rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=()) for _ in range(10)]
        expected_error = self.get_timeout_error_regex()
        for fut in futs:
            with self.assertRaisesRegex(RuntimeError, expected_error):
                fut.wait()
        rpc._set_rpc_timeout(200)
        fut1 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        rpc._set_rpc_timeout(0.001)
        fut2 = rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut2.wait()
        fut1.wait()
        rpc._set_rpc_timeout(0)
        rpc.rpc_async(worker_name(dst_rank), my_sleep_func, args=()).wait()
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init
    def test_rpc_timeouts(self):
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst_rank)
        timeout = 0.1
        expected_error = self.get_timeout_error_regex()
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,)).wait()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=timeout)
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_sleep_func, args=(1,))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,))
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=5).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=5)
        rpc.rpc_async(dst_worker, my_sleep_func, args=(1,), timeout=0).wait()
        rpc.rpc_sync(dst_worker, my_sleep_func, args=(1,), timeout=0)
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    def test_dist_init_decorator(self):

        @dist_init(setup_rpc=False)
        def test_func(self):
            return 'expected result'
        self.assertEqual(test_func(self), 'expected result')

        @dist_init
        def test_func(self):
            return 'expected result'
        self.assertEqual(test_func(self), 'expected result')

    def test_use_rpc_pickler(self):

        class TestPickler:
            pass
        test_pickler = TestPickler()
        with _use_rpc_pickler(test_pickler):
            self.assertTrue(torch.distributed.rpc.api._default_pickler is test_pickler)
        self.assertTrue(torch.distributed.rpc.api._default_pickler is _internal_rpc_pickler)

    @dist_init
    def test_wait_all(self):
        with _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            fut = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
            self.assertTrue(len(_thread_local_var.future_list) == 1)
            self.assertTrue(isinstance(_thread_local_var.future_list[0], torch._C.Future))
        self.assertTrue(fut.done())
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertFalse(hasattr(_thread_local_var, 'future_list'))

    @dist_init
    def test_wait_all_multiple_call(self):
        with _wait_all():
            self.assertTrue(_thread_local_var.future_list == [])
            dst = worker_name((self.rank + 1) % self.world_size)
            for i in range(20):
                fut = rpc.rpc_async(dst, torch.add, (torch.ones(i, i), 1))
                res = rpc.rpc_sync(dst, torch.add, (torch.ones(i, i), 1))
                self.assertEqual(res, torch.ones(i, i) + 1)
                self.assertEqual(fut.wait(), torch.ones(i, i) + 1)
            self.assertTrue(len(_thread_local_var.future_list) == 20)
        self.assertFalse(hasattr(_thread_local_var, 'future_list'))

    @dist_init
    def test_wait_all_timeout(self):
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            with _wait_all():
                self.assertTrue(_thread_local_var.future_list == [])
                dst = worker_name((self.rank + 1) % self.world_size)
                timeout = 0.1
                fut = rpc.rpc_async(dst, my_sleep_func, args=(1,), timeout=timeout)
        self.assertFalse(hasattr(_thread_local_var, 'future_list'))

    @dist_init
    def test_wait_all_raise_in_user_func(self):
        with self.assertRaises(ValueError):
            with _wait_all():
                self.assertTrue(_thread_local_var.future_list == [])
                dst = worker_name((self.rank + 1) % self.world_size)
                fut = rpc.rpc_async(dst, raise_func)
        self.assertFalse(hasattr(_thread_local_var, 'future_list'))

    @dist_init
    def test_wait_all_raise_in_body(self):
        with self.assertRaises(ValueError):
            with _wait_all():
                raise_func()
        self.assertFalse(hasattr(_thread_local_var, 'future_list'))

    @dist_init
    def test_custom_exception_throw_during_reconstruction(self):
        """
        Test that we still throw info about the remote side exception even when
        we cannot recreate it on client side.
        """
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank != 0:
            exc_caught = False
            dst = worker_name(0)
            try:
                rpc.rpc_sync(dst, custom_raise_func, args=())
            except RuntimeError as e:
                exc_caught = True
                msg = str(e)
                print(f'Got msg {msg}')
                self.assertTrue('Original exception on remote side was' in msg)
                self.assertTrue('CustomException' in msg)
            except BaseException as e:
                raise RuntimeError(f'Failure - expected RuntimeError, got {e}') from e
            finally:
                self.assertTrue(exc_caught)
        dist.barrier()
    timed_out_rpc_event = None

    @staticmethod
    def timed_out_rpc():
        RpcTest.timed_out_rpc_event.wait()

    @dist_init
    def test_wait_all_exit_early_python(self):
        RpcTest.timed_out_rpc_event = Event()
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        dst = worker_name((self.rank + 1) % self.world_size)
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, raise_func)
        fut3 = rpc.rpc_async(dst, raise_func)
        with self.assertRaisesRegex(ValueError, expected_err):
            torch.futures.wait_all([fut1, fut2, fut3])
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_wait_all_exit_early_builtin(self):
        RpcTest.timed_out_rpc_event = Event()
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        dst = worker_name((self.rank + 1) % self.world_size)
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, torch.add, args=(torch.rand(10), torch.rand(5)))
        fut3 = rpc.rpc_async(dst, torch.add, args=(torch.rand(10), torch.rand(5)))
        with self.assertRaisesRegex(RuntimeError, 'size of tensor'):
            torch.futures.wait_all([fut1, fut2, fut3])
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_wait_all_exit_early_script_function(self):
        RpcTest.timed_out_rpc_event = Event()
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        dst = worker_name((self.rank + 1) % self.world_size)
        fut1 = rpc.rpc_async(dst, RpcTest.timed_out_rpc)
        fut2 = rpc.rpc_async(dst, raise_func_script, args=(expected_err,))
        fut3 = rpc.rpc_async(dst, raise_func_script, args=(expected_err,))
        with self.assertRaisesRegex(RuntimeError, expected_err):
            torch.futures.wait_all([fut1, fut2, fut3])
        RpcTest.timed_out_rpc_event.set()

    @dist_init
    def test_function_not_on_callee(self):
        this_module = sys.modules[__name__]
        caller_worker = 'worker0'
        callee_worker = 'worker1'
        if self.rank == 1:
            delattr(this_module, 'foo_add')
            rpc.rpc_sync(caller_worker, set_value, args=(self.rank,))
        if self.rank == 0:
            wait_for_value_future()
            self.assertTrue(hasattr(this_module, 'foo_add'))
            with self.assertRaisesRegex(RuntimeError, 'RPC pickler does not serialize'):
                rpc.rpc_sync(callee_worker, foo_add, args=())

    @dist_init
    def test_non_garbage_collected_user_rref_due_to_local_circular_dependency(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        a = MyClass(1)
        b = MyClass(2)
        a.other = b
        b.other = a
        n = self.rank
        a.rref = rpc.remote(dst_worker_name, torch.add, args=(torch.ones(n, n), 2))

    @dist_init(setup_rpc=False)
    def test_use_rref_after_shutdown(self):
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        rpc.shutdown(graceful=True)
        with self.assertRaisesRegex(RuntimeError, 'Cannot call to_here\\(\\) on it after deletion.'):
            rref.to_here()
        with self.assertRaisesRegex(RuntimeError, 'Cannot call fork an UserRRef after deletion.'):
            import torch.distributed.rpc.internal as internal
            internal.serialize(rref)

    @staticmethod
    def _return_gpu_tensor():
        return torch.rand(3, 3).cuda(0)

    @staticmethod
    def _return_gpu_tensor_list():
        return [torch.rand(3, 3).cuda(0), torch.rand(3, 3).cuda(1)]

    @staticmethod
    def _gpu_tensor_list_arg(tensor_list):
        return torch.rand(3, 3)

    def _create_rref(self):
        owner_rank = (self.rank + 2) % self.world_size
        return rpc.remote(worker_name(owner_rank), torch.add, args=(torch.zeros(2, 2), 1))

    @dist_init
    def test_user_rrefs_confirmed(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret = rpc.rpc_sync(worker_name(dst_rank), check_rref_confirmed, args=(rref,))
        self.assertEqual(ret, True)

    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret_rref = rpc.remote(worker_name(dst_rank), check_rref_confirmed, args=(rref,))
        self.assertEqual(ret_rref.to_here(), True)

    @dist_init
    def test_rref_py_pickle_not_supported(self):
        local_rref = RRef(35)
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, 'Can not pickle rref in python pickler'):
                torch.save(local_rref, fname)

    @dist_init
    def test_remote_throw(self):
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), raise_or_inc, args=(torch.ones(2),))
        with self.assertRaisesRegex(Exception, '.*Expected error.*'):
            rref.to_here()

    @dist_init
    def test_non_cont_tensors(self):
        if self.rank == 0:
            t = torch.rand(5, 5)
            t_view = t.narrow(1, 2, 2)
            self.assertFalse(t_view.is_contiguous())
            t_cont = t_view.contiguous()
            self.assertTrue(t_cont.is_contiguous())
            self.assertEqual(t_view, t_cont)
            next_rank = (self.rank + 1) % self.world_size
            t_ret = rpc.rpc_sync(worker_name(next_rank), non_cont_test, args=(t_view, t_cont))
            self.assertEqual(t_view, t_ret)
            self.assertFalse(t_ret.is_contiguous())

    @dist_init
    def test_callback_simple(self):
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1

        def callback(fut):
            ret = fut.wait()
            self.assertEqual(ret, torch.ones(n, n) * 2)
            set_by_cb.set_result(ret.clone() + 1)
        fut = rpc.rpc_async(worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        fut.then(callback)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        self.assertEqual(set_by_cb.result(), torch.ones(n, n) * 2 + 1)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_wrong_arg_num(self):
        set_by_cb = concurrent.futures.Future()
        n = self.rank + 1
        fut = rpc.rpc_async(worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        cb_fut = fut.then(my_function)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        with self.assertRaisesRegex(RuntimeError, 'my\\_function\\(\\) missing 2 required positional arguments'):
            cb_fut.wait()

    @dist_init
    def test_callback_wrong_arg_type(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        fut0 = rpc.rpc_async(dst, torch.add, args=(torch.ones(2, 2), 1))
        fut1 = fut0.then(lambda x: x + 1)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operand type\\(s\\) for \\+'):
            fut1.wait()

    @dist_init
    def test_callback_multi(self):
        num_cbs = 10
        n = self.rank + 1

        def callback(idx, fut):
            ret = fut.wait()
            self.assertEqual(ret, torch.ones(n, n) * 2)
            return ret + idx
        fut = rpc.rpc_async(worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        cb_futs = []
        for idx in range(num_cbs):
            cb_futs.append(fut.then(partial(callback, idx)))
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        for idx in range(num_cbs):
            self.assertEqual(cb_futs[idx].wait(), torch.ones(n, n) * 2 + idx)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_callback_chain(self):
        n = self.rank + 1
        dst = worker_name(n % self.world_size)

        def callback(fut):
            return fut.wait() + 1
        fut = rpc.rpc_async(worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), 1))
        num_cbs = 20
        for _ in range(num_cbs):
            fut = fut.then(callback)
        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_callback_in_rpc(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(dst1, add_use_future_cb, args=(dst2, torch.ones(2, 2), 1, 2))
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_callback_with_ret(self):
        dst = worker_name((self.rank + 1) % self.world_size)

        def callback(fut0):
            fut2 = rpc.rpc_async(dst, torch.add, args=(fut0.wait(), 1)).then(lambda fut1: fut1.wait() + 1)
            return fut2.wait()
        fut3 = rpc.rpc_async(dst, torch.add, args=(torch.ones(2, 2), 1)).then(callback)
        self.assertEqual(fut3.wait(), torch.ones(2, 2) + 3)

    @dist_init
    def test_callback_with_error(self):
        dst = worker_name((self.rank + 1) % self.world_size)

        def callback(fut0):
            with self.assertRaisesRegex(ValueError, 'Expected error'):
                fut0.wait()
            raise RuntimeError('Another expected error')
        fut1 = rpc.rpc_async(dst, raise_func).then(callback)
        with self.assertRaisesRegex(RuntimeError, 'Another expected error'):
            fut1.wait()

    @dist_init
    def test_callback_none(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments.'):
            rpc.rpc_async(dst, raise_func).then(None)

    @dist_init
    def test_add_done_callback(self):
        set_by_cb = False
        n = self.rank + 1

        def callback(fut):
            nonlocal set_by_cb
            fut.wait()
            set_by_cb = True
        fut = rpc.rpc_async(worker_name(n % self.world_size), torch.add, args=(torch.ones(n, n), torch.ones(n, n)))
        fut.add_done_callback(callback)
        fut_then = fut.then(lambda _: True)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)
        fut_then.wait()
        self.assertTrue(set_by_cb)
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_mark_future_twice(self):
        fut = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), torch.add, args=(torch.zeros(2, 2), 1))
        self.assertEqual(fut.wait(), torch.zeros(2, 2) + 1)
        with self.assertRaisesRegex(RuntimeError, 'Future can only be marked completed once'):
            fut.set_result(1)

    @dist_init
    def test_pickle_future(self):
        fut = torch.futures.Future()
        errMsg = 'Can not pickle torch.futures.Future'
        dst = worker_name((self.rank + 1) % self.world_size)
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.rpc_sync(dst, fail_on_fut, args=(fut,))
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.rpc_async(dst, fail_on_fut, args=(fut,))
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                rpc.remote(dst, fail_on_fut, args=(fut,))

    @dist_init
    def test_future_done(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, torch.add, args=(torch.zeros(2), 1))
        fut.wait()
        self.assertTrue(fut.done())

    @dist_init
    def test_future_done_exception(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        fut = rpc.rpc_async(dst, raise_func)
        with self.assertRaisesRegex(ValueError, 'Expected error'):
            fut.wait()
        self.assertTrue(fut.done())

    def _test_future_cb(self, func):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(dst1, func, args=(dst2, torch.ones(2, 2), 1, 2))
        self.assertEqual(ret, torch.ones(2, 2) + 1 + 2)

    @dist_init
    def test_future_in_rpc(self):
        self._test_future_cb(add_use_future_set_result)

    @dist_init
    def test_future_nested_callback(self):
        self._test_future_cb(add_use_future_nested_cb)

    def _test_async_function_raise(self, mode):
        with self.assertRaisesRegex(RuntimeError, 'Expected error'):
            self._run_func_in_mode(worker_name((self.rank + 1) % self.world_size), async_raise_func, mode)

    @dist_init
    def test_async_function_raise(self):
        self._test_async_function_raise(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_raise_async(self):
        self._test_async_function_raise(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_raise_remote(self):
        self._test_async_function_raise(RPCExecMode.REMOTE)

    def _test_async_function_wrong_return_type(self, mode):
        errMsg = 'Functions decorated with @rpc\\.async_function must return a torch\\.futures\\.Future object,'
        with self.assertRaisesRegex(RuntimeError, errMsg):
            self._run_func_in_mode(worker_name((self.rank + 1) % self.world_size), async_wrong_type, mode)

    @dist_init
    def test_async_function_wrong_return_type(self):
        self._test_async_function_wrong_return_type(RPCExecMode.SYNC)

    @dist_init
    def test_async_function_wrong_return_type_async(self):
        self._test_async_function_wrong_return_type(RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        self._test_async_function_wrong_return_type(RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_simple(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(dst1, async_add, args=(dst2, torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    def _test_async_function(self, fn, mode=RPCExecMode.SYNC):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        args = (dst2, torch.ones(2, 2), 1, 2)
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + 3)

    @dist_init
    def test_async_function_with_future_ctor(self):
        self._test_async_function(async_add_with_future_ctor)

    @dist_init
    def test_async_function_with_future_ctor_remote(self):
        self._test_async_function(async_add_with_future_ctor, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_chained(self):
        self._test_async_function(async_add_chained)

    @dist_init
    def test_async_function_chained_remote(self):
        self._test_async_function(async_add_chained, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_nested(self):
        self._test_async_function(async_add_nested)

    @dist_init
    def test_async_function_nested_remote(self):
        self._test_async_function(async_add_nested, RPCExecMode.REMOTE)

    @dist_init
    def test_async_static_method(self):
        self._test_async_function(AsyncExecutionClass.static_async_add)

    @dist_init
    def test_async_static_method_remote(self):
        self._test_async_function(AsyncExecutionClass.static_async_add, RPCExecMode.REMOTE)

    @dist_init
    def test_async_class_method(self):
        self._test_async_function(AsyncExecutionClass.class_async_add)

    @dist_init
    def test_async_class_method_remote(self):
        self._test_async_function(AsyncExecutionClass.class_async_add, RPCExecMode.REMOTE)

    def _test_test_async_class_rref_proxy(self, mode=RPCExecMode.SYNC):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        rref = rpc.remote(dst1, AsyncExecutionClass)
        x = torch.ones(2, 2)
        y = torch.ones(2, 2) + 1
        if mode == RPCExecMode.SYNC:
            ret = rref.rpc_sync().static_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().class_async_add(dst2, x, x, y)
            ret += rref.rpc_sync().bound_async_add(dst2, x, x, y)
        elif mode == RPCExecMode.ASYNC:
            ret = rref.rpc_async().static_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().class_async_add(dst2, x, x, y).wait()
            ret += rref.rpc_async().bound_async_add(dst2, x, x, y).wait()
        elif mode == RPCExecMode.REMOTE:
            ret = rref.remote().static_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().class_async_add(dst2, x, x, y).to_here()
            ret += rref.remote().bound_async_add(dst2, x, x, y).to_here()
        self.assertEqual(ret, 3 * 4 * x)

    @dist_init
    def test_async_class_rref_proxy(self):
        self._test_test_async_class_rref_proxy()

    @dist_init
    def test_async_class_rref_proxy_async(self):
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.ASYNC)

    @dist_init
    def test_async_class_rref_proxy_remote(self):
        self._test_test_async_class_rref_proxy(mode=RPCExecMode.REMOTE)

    def _test_async_function_multi(self, fn, mode=RPCExecMode.SYNC):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        num = 20
        step = 3
        args = (dst2, torch.ones(2, 2), num, step)
        ret = self._run_func_in_mode(dst1, fn, mode, args=args)
        self.assertEqual(ret, torch.ones(2, 2) + num * step)

    @dist_init
    def test_async_function_multi_chained(self):
        self._test_async_function_multi(async_add_chained_multi)

    @dist_init
    def test_async_function_multi_chained_async(self):
        self._test_async_function_multi(async_add_chained_multi, RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_multi_chained_remote(self):
        self._test_async_function_multi(async_add_chained_multi, RPCExecMode.REMOTE)

    @dist_init
    def test_async_function_multi_fanout(self):
        self._test_async_function_multi(async_add_multi_fanout)

    @dist_init
    def test_async_function_multi_fanout_async(self):
        self._test_async_function_multi(async_add_multi_fanout, RPCExecMode.ASYNC)

    @dist_init
    def test_async_function_multi_fanout_remote(self):
        self._test_async_function_multi(async_add_multi_fanout, RPCExecMode.REMOTE)

    def _test_return_future(self, mode):
        with self.assertRaisesRegex(RuntimeError, 'Can not pickle torch.futures.Future'):
            self._run_func_in_mode(worker_name((self.rank + 1) % self.world_size), return_future, mode)

    @dist_init
    def test_return_future(self):
        self._test_return_future(RPCExecMode.SYNC)

    @dist_init
    def test_return_future_async(self):
        self._test_return_future(RPCExecMode.ASYNC)

    @dist_init
    def test_return_future_remote(self):
        self._test_return_future(RPCExecMode.REMOTE)

    @dist_init
    def test_rref_timeout(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, my_sleep_func, args=(2,), timeout=0.01)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rref.to_here()
        wait_until_owners_and_forks_on_rank(1, 1, rank=1)

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(os.environ.get('RPC_INIT_WITH_TCP', None) == '1', 'init_pg_then_rpc does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614.')
    def test_init_pg_then_rpc(self):
        dist.init_process_group(backend='gloo', init_method=self.init_method, rank=self.rank, world_size=self.world_size)
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)
        dist.barrier()
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(os.environ.get('RPC_INIT_WITH_TCP', None) == '1', 'init_rpc_then_pg does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614.')
    def test_init_rpc_then_pg(self):
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        dist.init_process_group(backend='gloo', init_method=self.init_method, rank=self.rank, world_size=self.world_size)
        next_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(ret, torch.ones(2, 2) + 1)
        dist.barrier()
        rpc.shutdown()

    @dist_init
    def test_wait_all_with_exception(self):
        futs = []
        dst = worker_name((self.rank + 1) % self.world_size)
        for _ in range(10):
            futs.append(rpc.rpc_async(dst, raise_func))
        with self.assertRaisesRegex(ValueError, 'Expected error'):
            ret = torch.futures.wait_all(futs)

    @dist_init
    def test_wait_all_with_partial_exception(self):
        futs = []
        dst = worker_name((self.rank + 1) % self.world_size)
        for _ in range(10):
            futs.append(rpc.rpc_async(dst, torch.add, args=(torch.ones(2), 1)))
        futs.append(rpc.rpc_async(dst, raise_func))
        with self.assertRaisesRegex(ValueError, 'Expected error'):
            ret = torch.futures.wait_all(futs)

    @dist_init(setup_rpc=False)
    @skip_but_pass_in_sandcastle_if(os.environ.get('RPC_INIT_WITH_TCP', None) == '1', 'Test does not work with TCP init, see https://github.com/pytorch/pytorch/issues/46491')
    def test_init_rpc_twice(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        rpc.shutdown()
        dist.barrier()
        new_backend_options = self.rpc_backend_options
        new_backend_options.init_method += 'init_2'
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=new_backend_options)
        dst = worker_name((self.rank + 1) % self.world_size)
        rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))
        rpc.rpc_sync(dst, foo_add, args=())
        rpc.shutdown()

    def test_wrong_types(self):
        with self.assertRaisesRegex(TypeError, 'Argument backend must be a member of BackendType'):
            rpc.init_rpc(name=worker_name(self.rank), rank=self.rank, world_size=self.world_size, backend='TENSORPIPE')
        with self.assertRaisesRegex(TypeError, 'Argument rpc_backend_options must be an instance of RpcBackendOptions'):
            rpc.init_rpc(name=worker_name(self.rank), rank=self.rank, world_size=self.world_size, backend=self.rpc_backend, rpc_backend_options={'init_method': self.init_method})

    def test_cannot_infer_backend_from_options(self):
        rpc_backend_options = FooBackendOptions(self.init_method)
        with self.assertRaisesRegex(TypeError, 'Could not infer backend for options'):
            rpc.init_rpc(name=worker_name(self.rank), rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)

    @dist_init
    def test_owner_rref_backward(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        t1 = torch.rand(10, 10, requires_grad=True)
        rref = rpc.RRef(t1.sum() + t1.sum())
        rref.backward()
        expected_grad = torch.ones_like(t1) * 2
        self.assertEqual(expected_grad, t1.grad)
        with dist_autograd.context() as context_id:
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            rref = rpc.RRef(t2.sum())
            rref.backward(context_id)
            self.assertEqual(expected_grad, dist_autograd.get_gradients(context_id)[t1])
        with dist_autograd.context() as context_id:
            t2 = rpc.rpc_sync(dst, torch.add, args=(t1, t1))
            rref = rpc.RRef(t2.sum())
            rref.backward(context_id, retain_graph=True)
            rref.backward(context_id)
            self.assertEqual(expected_grad * 2, dist_autograd.get_gradients(context_id)[t1])
        with self.assertRaisesRegex(RuntimeError, 'tensors does not require grad and does not have a grad_fn'):
            rpc.RRef(torch.rand(10)).backward()
        with self.assertRaisesRegex(RuntimeError, 'grad can be implicitly created only for scalar outputs'):
            rpc.RRef(torch.rand(10, requires_grad=True)).backward()
        with self.assertRaisesRegex(RuntimeError, 'Could not find autograd context with id: 100'):
            rpc.RRef(torch.rand(10, requires_grad=True).sum()).backward(100)
        with self.assertRaisesRegex(RuntimeError, 'RRef should contain a tensor for .backward()'):
            rpc.RRef('foo').backward()

    @staticmethod
    def _sum(x):
        return x.sum()

    @staticmethod
    def _identity(x):
        return x

    @dist_init
    def test_user_rref_backward(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        t = torch.rand(10, requires_grad=True)
        with dist_autograd.context() as context_id:
            rref = rpc.remote(dst, RpcTest._sum, args=(t,))
            rref.backward(context_id, retain_graph=True)
            rref.backward(context_id)
            self.assertEqual(torch.ones_like(t) * 2, dist_autograd.get_gradients(context_id)[t])
        with dist_autograd.context() as context_id:
            rref = rpc.remote(dst, RpcTest._identity, args=('foo',))
            with self.assertRaisesRegex(RuntimeError, 'RRef should contain a tensor for .backward()'):
                rref.backward(context_id)
            with self.assertRaisesRegex(RuntimeError, "User RRefs require 'dist_autograd_ctx_id' to be specified"):
                rref.backward()

    @dist_init(setup_rpc=False)
    def test_shutdown_errors(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        if self.rank != 0:
            og_func = rpc.api._broadcast_to_followers
            og_rref_func = rpc.api._delete_all_user_and_unforked_owner_rrefs

            def raise_error(sequence_id, objects_map):
                og_func(sequence_id, objects_map)
                raise RuntimeError('simulation')

            def rref_error():
                raise RuntimeError('simulation rref')
            try:
                rpc.api._broadcast_to_followers = raise_error
                rpc.api._delete_all_user_and_unforked_owner_rrefs = rref_error
                with self.assertRaisesRegex(RuntimeError, 'simulation rref'):
                    rpc.shutdown()
            finally:
                rpc.api._broadcast_to_followers = og_func
                rpc.api._delete_all_user_and_unforked_owner_rrefs = og_rref_func
        else:
            with self.assertRaisesRegex(RuntimeError, 'timed out in _all_gather'):
                rpc.shutdown()
        dist.barrier()

    @dist_init
    def test_my_parameter_server(self):
        self._my_parameter_server(False)