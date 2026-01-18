from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
class JitFaultyAgentRpcTest(RpcAgentTestFixture):
    """
    Run tests for rpc_async in JIT under the faulty agent test fixture to test
    arbitrary timeouts.
    """

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_CALL': 1.5})
    def test_timeout_in_torchscript_function(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {'first_kwarg': torch.tensor([2, 2]), 'second_kwarg': torch.tensor([3, 3])}
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0.5)
        rpc._set_rpc_timeout(0.001)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            script_rpc_async_call(dst_worker_name, args, kwargs)
        ret = rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0)
        self.assertEqual(ret, torch.tensor([8, 8]))
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_CALL': 1.5})
    def test_timeout_in_python(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {'first_kwarg': torch.tensor([2, 2]), 'second_kwarg': torch.tensor([3, 3])}
        expected_error = self.get_timeout_error_regex()
        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0.5)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        rpc._set_rpc_timeout(0.001)
        fut = rpc_async_call_future_ret(dst_worker_name, args, kwargs)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0)
        result = fut.wait()
        self.assertEqual(result, torch.tensor([8, 8]))
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
    def test_remote_timeout_to_here_in_jit(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rref_to_here(rref)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_RREF_FETCH_CALL': 1})
    def test_rref_to_here_timeout_in_jit(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref_to_here_with_timeout(rref, 0.01)
        rref_to_here_with_timeout(rref, 100)

    @dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
    def test_rref_timeout_pickle_in_jit(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rpc_async_with_rref_arg(dst_worker, (rref,))

    @dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
    def test_rref_timeout_pickle_script_func(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rpc.rpc_sync(dst_worker, rref_to_here, args=(rref,))