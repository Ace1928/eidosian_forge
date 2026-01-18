import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
class DistAutogradTest(CommonDistAutogradTest):

    @dist_init
    def test_autograd_context(self):
        max_auto_increment = 281474976710655
        self.assertEqual(max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id())
        context_ids = []
        for i in range(200):
            with dist_autograd.context() as context_id:
                self.assertEqual(context_id, dist_autograd._retrieve_context(context_id)._context_id())
                self.assertEqual(self.worker_id, context_id >> 48)
                context_ids.append(context_id)
        for context_id in context_ids:
            with self.assertRaisesRegex(RuntimeError, f'Could not find autograd context with id: {context_id}'):
                dist_autograd._retrieve_context(context_id)

    @dist_init
    def test_nested_context(self):
        with dist_autograd.context() as context_id:
            with self.assertRaisesRegex(RuntimeError, 'Already have an autograd context id for this thread'):
                with dist_autograd.context() as context_id:
                    pass

    @dist_init
    def test_graph_for_builtin_call(self):
        self._test_graph(torch.add, ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_python_call(self):
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_builtin_remote_call(self):
        self._test_graph(torch.add, ExecMode.REMOTE, False)

    @dist_init
    def test_graph_for_python_remote_call(self):
        self._test_graph(my_py_add, ExecMode.REMOTE, False)

    @dist_init
    def test_graph_for_py_nested_call(self):
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_py_nested_remote_call(self):
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, False)

    @dist_init
    def test_graph_for_py_nested_call_itself(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_graph_for_py_nested_remote_call_itself(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, False)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, False)

    def _test_grad_only_on_return_value(self, exec_mode):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), ret_requires_grad)
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), ret_requires_grad).to_here()
            else:
                raise ValueError(f'Unrecognized ExecMode {exec_mode}')
            dist_autograd.backward(context_id, [ret.sum()])
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            self._check_rpc_done(1)
            grads = dist_autograd.get_gradients(ctx_ids[1])
            self.assertEqual(1, len(grads))
            self.assertIn(requires_grad_tensor, grads)
            self.assertEqual(torch.ones_like(ret), grads[requires_grad_tensor])
            dist.barrier()

    @dist_init
    def test_grad_only_on_return_value(self):
        self._test_grad_only_on_return_value(ExecMode.RPC_SYNC)

    @dist_init
    def test_grad_only_on_return_value_remote(self):
        self._test_grad_only_on_return_value(ExecMode.REMOTE)

    @dist_init
    def test_rpc_complex_args(self):
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, False)

    @dist_init
    def test_remote_complex_args(self):
        self._test_rpc_complex_args(ExecMode.REMOTE, False)

    @dist_init
    def test_context_cleanup_tensor_with_grad(self):
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_context_cleanup_tensor_no_grad(self):
        t1 = torch.ones(3, 3, requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    def test_context_cleanup_no_tensors(self):
        self.context_cleanup_test_helper(rpc_args=(1, 1), func=my_scalar_add)

    @dist_init
    def test_context_cleanup_nested_rpc(self):
        t1 = torch.ones(3, 3, requires_grad=True)
        t2 = torch.zeros(3, 3, requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(rpc_args=args, func=my_py_nested_call, nested=True)

    @dist_init
    def test_worker_ids_recorded(self):
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)
            t1.requires_grad = True
            t2.requires_grad = True
            for dst_rank in dst_ranks:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(worker_ids, dst_ranks)

    @dist_init
    def test_dist_autograd_profiling(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(3, 3, requires_grad=True)
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2)).sum()
            with torch.autograd.profiler.profile() as p:
                dist_autograd.backward(context_id, [loss])
        function_events = p.function_events

        def get_event(partial_key):
            return next((event for event in function_events if partial_key in event.name))
        send_event = get_event('SendRpcBackward')
        recv_event = get_event('RecvRpcBackward')
        backward_event = get_event('torch::distributed::autograd::backward')
        self.assertEqual(send_event.count, 1)
        self.assertEqual(recv_event.count, 1)
        self.assertGreater(backward_event.cpu_time_total, send_event.cpu_time_total)
        self.assertGreater(backward_event.cpu_time_total, recv_event.cpu_time_total)

    @dist_init
    def test_error_in_context(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(6, 6, requires_grad=True)
            with self.assertRaises(RuntimeError):
                rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(t1, t2))

    @dist_init
    def test_backward_no_grad_on_tensor(self):
        self._backward_no_grad_on_tensor(torch.rand((3, 3), requires_grad=True), torch.rand((3, 3), requires_grad=True), False)

    @dist_init
    def test_backward_simple(self):
        self._backward_simple(self._next_rank(), torch.rand((3, 3), requires_grad=True), torch.rand((3, 3), requires_grad=True), None, False)

    @dist_init
    def test_backward_simple_self(self):
        self._backward_simple(self.rank, torch.rand((3, 3), requires_grad=True), torch.rand((3, 3), requires_grad=True), None, False)

    @dist_init
    def test_backward_rref(self):
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._backward_rref(callee, rref_owner, torch.rand((3, 3), requires_grad=True), torch.rand((3, 3), requires_grad=True), None, False)

    @dist_init
    def test_backward_rref_multi(self):
        if self.rank > 0:
            callee = 'worker0'
            rref_owner = callee
            self._backward_rref(callee, rref_owner, torch.rand((3, 3), requires_grad=True), torch.rand((3, 3), requires_grad=True), None, False)

    @dist_init
    def test_backward_rref_nested(self):
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._backward_rref(callee, rref_owner, torch.rand((3, 3), requires_grad=True), torch.rand((3, 3), requires_grad=True), None, False)

    @dist_init
    def test_trainer_ps(self):
        self._test_trainer_ps(create_tensor, _run_trainer, False)

    @dist_init
    def test_trainer_ps_torchscript_functions(self):
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = True
        self._test_trainer_ps(create_torchscript_tensor, _run_trainer_torchscript, False)

    @dist_init
    def test_backward_multiple_round_trips(self):
        self._backward_multiple_round_trips(torch.rand((3, 3), requires_grad=True), torch.rand((3, 3)), torch.rand((3, 3), requires_grad=True), torch.rand((3, 3)), torch.rand((3, 3), requires_grad=True), None, False)

    @dist_init
    def test_backward_different_tensor_dims(self):
        local_grads = None
        t1 = torch.rand((4, 6), requires_grad=True)
        t2 = torch.rand((6, 5))
        t3 = torch.rand((5, 7), requires_grad=True)
        t4 = torch.rand((7, 9))
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.matmul, t1, t2)
                val = self._exec_func(exec_mode, torch.linalg.multi_dot, (val, t3, t4))
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t2, t3, t4)
                local_grads = ret if ret else local_grads

    @dist_init
    def test_backward_unused_tensors(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        t3 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                s = self._exec_func(exec_mode, torch.stack, (t1, t2, t3))
                val = self._exec_func(exec_mode, torch.matmul, torch.narrow(s, 0, 0, 1), torch.narrow(s, 0, 2, 1))
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3)
                local_grads = ret if ret else local_grads

    @dist_init
    def test_backward_multiple_output_tensors(self):
        local_grads = None
        t = torch.rand((10, 2), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                tensor_list = self._exec_func(exec_mode, torch.split, t, 2)
                t1 = tensor_list[0]
                t2 = tensor_list[2]
                t3 = tensor_list[4]
                val = self._exec_func(exec_mode, torch.linalg.multi_dot, (t1, t2, t3))
                loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t)
                local_grads = ret if ret else local_grads

    def _run_test_backward_unused_send_function_in_thread(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            val = torch.mul(t1, t2)
            dist_autograd.backward(context_id, [val.sum()])

    @dist_init
    def test_backward_unused_send_function(self):
        t = threading.Thread(target=self._run_test_backward_unused_send_function_in_thread)
        t.daemon = True
        t.start()
        t.join(10)
        self.assertTrue(t.is_alive())

    @dist_init
    def test_backward_autograd_engine_error(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            tmp = (t1 + t2) * (t1 + t2)
            t3 = SimulateBackwardError.apply(tmp)
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t2, t3))
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.mul, args=(val, t2))
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(val, t2))
            val = rpc.rpc_sync(worker_name(self._next_rank()), torch.div, args=(val, t2))
            with self.assertRaisesRegex(RuntimeError, 'Error on Node [0-9]+: Simulate error on backward pass'):
                dist_autograd.backward(context_id, [val.sum()])

    @dist_init(clean_shutdown=False)
    @skip_but_pass_in_sandcastle_if(IS_MACOS, 'Test is flaky on MacOS since libuv error handling is not as robust as TCP')
    def test_backward_node_failure(self):
        rpc._set_rpc_timeout(5)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            dist.barrier()
            if self.rank % 2 == 0:
                shutdown_error_regex = self.get_shutdown_error_regex()
                for rank in range(self.world_size):
                    if rank % 2 != 0:
                        wait_until_node_failure(rank, shutdown_error_regex)
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    dist_autograd.backward(context_id, [res.sum()])
            else:
                pass

    @dist_init
    def test_backward_without_context(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        context_id = 100
        with self.assertRaisesRegex(RuntimeError, f'Could not find autograd context with id: {context_id}'):
            res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            dist_autograd.backward(context_id, [res.sum()])

    @dist_init
    def test_backward_without_rpc(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.add(t1, t2)
            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init
    def test_backward_invalid_args(self):
        with dist_autograd.context() as context_id:
            with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
                dist_autograd.backward(context_id, None)
            with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
                dist_autograd.backward(None, None)
            with self.assertRaisesRegex(RuntimeError, 'No tensors provided for gradient computation'):
                dist_autograd.backward(context_id, [])
            with self.assertRaisesRegex(RuntimeError, 'requires_grad not set on'):
                t = torch.rand(3, 3)
                dist_autograd.backward(context_id, [t])
            with self.assertRaisesRegex(RuntimeError, 'is not a scalar, all roots need to be scalar'):
                t = torch.rand(3, 3, requires_grad=True)
                dist_autograd.backward(context_id, [t])
            with self.assertRaisesRegex(RuntimeError, 'does not have a valid gradient function'):
                t = torch.rand(1, requires_grad=True)
                dist_autograd.backward(context_id, [t])

    @dist_init
    def test_backward_multiple_roots(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
            with dist_autograd.context() as context_id:
                r1 = self._exec_func(exec_mode, torch.add, t1, t2).sum()
                r2 = self._exec_func(exec_mode, torch.mul, t1, t2).sum()
                r3 = self._exec_func(exec_mode, torch.cos, t1).sum()
                r4 = self._exec_func(exec_mode, torch.div, t1, t2).sum()
                local_grads = self._verify_backwards(exec_mode, [r1, r2, r3, r4], context_id, local_grads, t1, t2)

    @dist_init
    def test_backward_different_dtypes(self):
        self._backward_different_dtypes(torch.rand((3, 3), requires_grad=True, dtype=torch.float32), torch.rand((3, 3), requires_grad=True, dtype=torch.float64), False)

    @dist_init
    def test_backward_simple_python_udf(self):
        self._backward_simple_python_udf(torch.rand(3, 3, requires_grad=True), torch.rand(3, 3, requires_grad=True), False)

    @dist_init
    def test_backward_simple_script_call(self):
        self._backward_simple_script_call(torch.rand(3, 3, requires_grad=True), torch.rand(3, 3, requires_grad=True), False)

    @staticmethod
    def _complex_python_udf(t1, t2):
        t3 = torch.nn.functional.linear(t1, t2)
        t4 = torch.nn.functional.linear(t2, t3)
        t5 = torch.nn.functional.linear(t3, t4)
        return torch.linalg.multi_dot([t1, t2, t3, t4, t5])

    @dist_init
    def test_backward_complex_python_udf(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, DistAutogradTest._complex_python_udf, t1, t2)
                loss = ret.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)

    @staticmethod
    def _python_udf_with_backward_error(t1, t2):
        t3 = t1 + t2
        t4 = SimulateBackwardError.apply(t3)
        return torch.linalg.multi_dot([t1, t2, t3, t4])

    @staticmethod
    def _nested_rpc_call_backward_error(t1, t2, dst):
        t1 = t1 * t2
        t2 = t1 + t2
        res = rpc.rpc_sync(worker_name(dst), DistAutogradTest._python_udf_with_backward_error, args=(t1, t2))
        return torch.linalg.multi_dot([t1, t2, res])

    @dist_init
    def test_backward_python_udf_error(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._nested_rpc_call_backward_error, args=(t1, t2, self._next_rank()))
            with self.assertRaisesRegex(RuntimeError, 'Simulate error on backward pass'):
                dist_autograd.backward(context_id, [loss.sum()])
    _backward_done = False

    @dist_init(clean_shutdown=False)
    @skip_but_pass_in_sandcastle_if(IS_MACOS, 'Test is flaky on MacOS since libuv error handling is not as robust as TCP')
    def test_backward_node_failure_python_udf(self):
        rpc._set_rpc_timeout(5)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            dst = self._next_rank()
            res = rpc.rpc_sync(worker_name(dst), my_py_nested_call, args=(t1, t2, dst, self.world_size, 1))
            dist.barrier()
            if self.rank == 2:
                return
            store = dist.distributed_c10d._get_default_store()
            if self.rank == 0:
                shutdown_error_regex = self.get_shutdown_error_regex()
                wait_until_node_failure(2, shutdown_error_regex)
                with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                    dist_autograd.backward(context_id, [res.sum()])
                store.set('test_backward_node_failure_python_udf_rank0_done', 'True')
            else:
                store.wait(['test_backward_node_failure_python_udf_rank0_done'], timedelta(seconds=10))

    @staticmethod
    def _nested_python_udf(t1, t2, dst):
        t3 = t1 * t2
        t4 = t1 + t2
        res = rpc.rpc_sync(worker_name(dst), my_py_add, args=(t3, t4))
        return t1 * t2 * t3 * t4 * res

    @dist_init
    def test_backwards_nested_python_udf(self):
        self._backwards_nested_python_udf(torch.rand(3, 3, requires_grad=True), torch.rand(3, 3, requires_grad=True), False)
    _test_clean_context_backward_context_id = None

    class MyBackwardFunc(Function):

        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            assert DistAutogradTest._test_clean_context_backward_context_id is not None
            dist.barrier()
            dist_autograd._release_context(DistAutogradTest._test_clean_context_backward_context_id)
            assert _all_contexts_cleaned_up()
            return input

    @dist_init
    def test_clean_context_during_backward(self):
        """
        This test simulates the situation where the 'backward' call might throw
        an exception locally which would lead to the autograd context being
        cleaned up if we're using the context manager. As a result, the autograd
        context might be cleaned up while some threads are still using the
        autograd context.

        It is fine for the 'backward' call to throw an exception in this test,
        but the process should not crash.
        """
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        context = dist_autograd._new_context()
        context_id = context._context_id()
        DistAutogradTest._test_clean_context_backward_context_id = context_id
        for i in range(0, self.world_size):
            if i != self.rank:
                rank_distance = (i - self.rank + self.world_size) % self.world_size
                rpc.rpc_sync(worker_name(i), _set_rpc_done, args=(context_id, rank_distance))
        dist.barrier()
        self.assertEqual(self.world_size - 1, len(known_context_ids))
        t1 = torch.rand((3, 3), requires_grad=True)
        for i in range(0, 100):
            dst = self._next_rank()
            t1 = rpc.rpc_sync(worker_name(dst), torch.add, args=(t1, t1))
        t1 = DistAutogradTest.MyBackwardFunc.apply(t1)
        self.assertEqual(100, len(context._send_functions()))
        context_id = 100
        with self.assertRaisesRegex(RuntimeError, f'Could not find autograd context with id: {context_id}'):
            dist_autograd.backward(context_id, [t1.sum()])
        dist.barrier()
        rpc.shutdown(graceful=False)
        sys.exit(0)

    @classmethod
    def _call_remote_embedding(cls, embedding_rref, input, offsets, per_sample_weights):
        embedding = embedding_rref.local_value()
        return embedding(input, offsets, per_sample_weights)

    @classmethod
    def _get_grad(cls, embedding_rref, context_id):
        embedding = embedding_rref.local_value()
        grad_map = dist_autograd.get_gradients(context_id)
        return grad_map[embedding.weight]

    @classmethod
    def _mixed_requires_grad_operaton(cls, t1, t2):
        if t2.requires_grad:
            return t1 - t2
        else:
            return t1 * t2

    @dist_init
    def test_mixed_requires_grad(self):
        self._mixed_requires_grad(torch.rand(3, 3, requires_grad=True), torch.rand(3, 3, requires_grad=False), False)

    class TestDebugInfoFunc(Function):

        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            debug_info = dist_autograd._get_debug_info()
            assert debug_info is not None
            backward_passes = int(debug_info['num_current_backward_passes'])
            assert backward_passes >= 1 and backward_passes <= 4
            return input

    @dist_init
    def test_debug_info(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            i = 0
            res = {}
            res[i] = t1
            for rank in range(self.world_size):
                if rank != self.rank:
                    res[i + 1] = rpc.rpc_sync(worker_name(rank), torch.add, args=(res[i], t2))
                    i += 1
            res[i + 1] = DistAutogradTest.TestDebugInfoFunc.apply(res[i])
            i += 1
            for rank in range(self.world_size):
                if rank != self.rank:
                    res[i + 1] = rpc.rpc_sync(worker_name(rank), torch.add, args=(res[i], t2))
                    i += 1
            dist_autograd.backward(context_id, [res[i].sum()])
            debug_info = dist_autograd._get_debug_info()
            num_autograd_context = int(debug_info['num_autograd_contexts'])
            self.assertTrue(num_autograd_context >= 1 and num_autograd_context <= 4)
        for rd in range(self.world_size - 1):
            rpc.rpc_sync(worker_name((self.rank + rd + 1) % self.world_size), _set_rpc_done, args=(context_id, rd + 1))
        dist.barrier()
        debug_info = dist_autograd._get_debug_info()
        assert debug_info is not None
        self.assertEqual(0, int(debug_info['num_current_backward_passes']))
        self.assertTrue(len(debug_info) == 2)
        self.assertTrue(_all_contexts_cleaned_up())
        debug_info = dist_autograd._get_debug_info()
        self.assertEqual(0, int(debug_info['num_autograd_contexts']))

    @staticmethod
    def _workload_thread():
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            t3 = rpc.rpc_sync('worker0', torch.add, args=(t1, t2))
            t4 = rpc.rpc_sync('worker0', torch.mul, args=(t2, t3))
            t5 = rpc.rpc_sync('worker0', torch.matmul, args=(t3, t4))
            t6 = rpc.rpc_sync('worker0', torch.add, args=(t4, t5))
            dist_autograd.backward(context_id, [t6.sum()])

    @dist_init
    def test_async_dist_autograd(self):
        """
        This test ensures async processing for distributed autograd works
        appropriately. This is achieved by spawning multiple threads and
        hammering a single node with a lot of backward() calls.
        """
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank != 0:
            threads = []
            for i in range(20):
                t = threading.Thread(target=DistAutogradTest._workload_thread)
                t.start()
                threads.append(t)
            for thread in threads:
                thread.join()
        dist.barrier()

    @dist_init
    def test_backward_accumulate_grads(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            t3 = torch.matmul(t1, t2)
            torch.autograd.backward([t3.sum()], retain_graph=True)
            torch.autograd.backward([t3.sum()])
            t3 = rpc.rpc_sync(worker_name(self._next_rank()), torch.matmul, args=(t1, t2))
            dist_autograd.backward(context_id, [t3.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    @staticmethod
    def _test_nested_backward_accumulate_grads(t1, t2, dst_rank):
        return rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))

    @dist_init
    def test_nested_backward_accumulate_grads(self):
        self._nested_backward_accumulate_grads(torch.rand(3, 3, requires_grad=True), torch.rand(3, 3, requires_grad=True), False)

    @dist_init
    def test_multiple_backward(self):
        self._multiple_backward(torch.rand(3, 3, requires_grad=True), torch.rand(3, 3, requires_grad=True), False)

    @dist_init(clean_shutdown=False)
    def test_multiple_backward_with_errors(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(f'worker{self._next_rank()}', DistAutogradTest._python_udf_with_backward_error, args=(t1, t2)).sum()
            try:
                for i in range(100):
                    if i < 50:
                        with self.assertRaisesRegex(RuntimeError, 'Simulate error on backward pass'):
                            dist_autograd.backward(context_id, [loss], retain_graph=True)
                    elif i > 50:
                        dist_autograd.backward(context_id, [loss], retain_graph=True)
                    else:
                        dist.barrier()
                        SimulateBackwardError._simulate_error = False
                        dist.barrier()
            finally:
                dist.barrier()
                SimulateBackwardError._simulate_error = True

    @dist_init
    def test_backward_verify_hooks(self):
        t1 = torch.ones((3, 3), requires_grad=True)
        t1.register_hook(lambda grad: grad * 2)
        t2 = torch.ones((3, 3), requires_grad=True)
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, torch.matmul, t1, t2)
                loss = ret.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
                local_grads = ret if ret else local_grads

    @dist_init
    def test_no_grad_copy(self):
        """
        Similar to test in test_autograd.py.
        """

        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return (grad, grad)

        class MyFuncSingleGrad(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp):
                return inp

            @staticmethod
            def backward(ctx, grad):
                MyFuncSingleGrad.static_grad_ptr = grad.data_ptr()
                return grad

        class NonContGradFunc(Function):

            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [NonContGradFunc.apply(MyFunc.apply(a, b))])
            grads = dist_autograd.get_gradients(context_id)
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFuncSingleGrad.apply(a)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFuncSingleGrad.static_grad_ptr
            p_a = grads[a].data_ptr()
            self.assertTrue(p_a == p_g)
        with dist_autograd.context() as context_id:
            dist_autograd.backward(context_id, [MyFunc.apply(a, b)[1][0]])
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a].data_ptr()
            p_b = grads[b].data_ptr()
            self.assertFalse(p_a == p_b)
            self.assertFalse(grads[a].data_ptr() == MyFunc.static_grad_ptr)
            self.assertFalse(grads[b].data_ptr() == MyFunc.static_grad_ptr)

    @dist_init
    def test_no_grad_copy_sparse(self):

        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp):
                return inp

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return (ngrad, ngrad)
        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F
        with dist_autograd.context() as context_id:
            emb_matrix = MyFunc.apply(a)
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a]._values().data_ptr()
            self.assertTrue(p_a == p_g)
            for i in range(10):
                dist_autograd.backward(context_id, [loss], retain_graph=True)
        with dist_autograd.context() as context_id:
            emb_matrix = NonContGradFunc.apply(a, b)
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            grads = dist_autograd.get_gradients(context_id)
            p_g = NonContGradFunc.static_grad_ptr
            p_a = grads[a]._values().data_ptr()
            p_b = grads[b]._values().data_ptr()
            self.assertFalse(p_a == p_b)
            self.assertFalse(p_a == p_g)
            self.assertFalse(p_b == p_g)
            for i in range(10):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

    @dist_init
    def test_grad_copy_sparse_indices_extra_ref(self):

        class MyFunc(Function):
            static_grad_ptr = None
            static_grad_indices_ref = None
            static_grad_values_ref = None

            @staticmethod
            def forward(ctx, inp):
                return inp

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                MyFunc.static_grad_indices_ref = grad._indices()
                MyFunc.static_grad_values_ref = grad._values()
                return grad
        a = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F
        with dist_autograd.context() as context_id:
            emb_matrix = MyFunc.apply(a)
            loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            grads = dist_autograd.get_gradients(context_id)
            p_g = MyFunc.static_grad_ptr
            p_a = grads[a]._values().data_ptr()
            self.assertIsNotNone(MyFunc.static_grad_indices_ref)
            self.assertIsNotNone(MyFunc.static_grad_values_ref)
            self.assertTrue(p_g == p_a)

    @dist_init
    def test_post_hooks(self):
        self.hook_called_times = 0

        def post_hook_add_one(output_grads, input_grads):
            self.hook_called_times += 1
            return output_grads

        def post_hook_add_two(output_grads, input_grads):
            self.hook_called_times += 2
            return output_grads
        t = torch.rand(10, 10, requires_grad=True)
        a = t + t
        accumulate_grad_0 = a.grad_fn.next_functions[0][0]
        accumulate_grad_0.register_hook(post_hook_add_one)
        accumulate_grad_0.register_hook(post_hook_add_two)
        accumulate_grad_1 = a.grad_fn.next_functions[1][0]
        accumulate_grad_1.register_hook(post_hook_add_two)
        with dist_autograd.context() as context_id:
            loss = a.sum()
            dist_autograd.backward(context_id, [loss])
            self.assertEqual(5, self.hook_called_times)
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(1, len(grads))
            self.assertTrue(t in grads)

    @staticmethod
    def _slow_add(t1, t2):
        time.sleep(1)
        t3 = t1 + t2
        t3.requires_grad = True
        return t3

    @dist_init
    def test_thread_local_context_id(self):
        t1 = torch.rand((3, 3))
        t2 = torch.rand((3, 3))
        t3 = t1 + t2
        t3.requires_grad = True
        t3.sum().backward()
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, DistAutogradTest._slow_add, args=(t1, t2))
        with dist_autograd.context() as context_id:
            loss = rref.to_here().sum()
            dist_autograd.backward(context_id, [loss])
            self.assertTrue(rpc.rpc_sync(dst, _compare_owner_value, args=(context_id, rref, t3.grad)))