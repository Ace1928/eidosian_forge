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
class TensorPipeAgentDistAutogradTest(CommonDistAutogradTest):

    @dist_init
    def test_graph_for_builtin_call_sparse(self):
        self._test_graph(torch.add, ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_python_call_sparse(self):
        self._test_graph(my_py_add, ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_builtin_remote_call_sparse(self):
        self._test_graph(torch.add, ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_python_remote_call_sparse(self):
        self._test_graph(my_py_add, ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_py_nested_call_sparse(self):
        self._test_graph_for_py_nested_call(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_py_nested_remote_call_sparse(self):
        self._test_graph_for_py_nested_call(ExecMode.REMOTE, True)

    @dist_init
    def test_graph_for_py_nested_call_itself_sparse(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_graph_for_py_nested_remote_call_itself_sparse(self):
        self._test_graph_for_py_nested_call_itself(ExecMode.REMOTE, True)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_sparse(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_no_graph_with_tensors_not_require_grad_remote_sparse(self):
        self._test_no_graph_with_tensors_not_require_grad(ExecMode.REMOTE, True)

    @dist_init
    def test_rpc_complex_args_sparse(self):
        self._test_rpc_complex_args(ExecMode.RPC_SYNC, True)

    @dist_init
    def test_remote_complex_args_sparse(self):
        self._test_rpc_complex_args(ExecMode.REMOTE, True)

    @dist_init
    def test_context_cleanup_tensor_with_grad_sparse(self):
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        self.context_cleanup_test_helper(rpc_args=(t1, t2), func=torch.add)

    @dist_init
    def test_context_cleanup_tensor_no_grad_sparse(self):
        t1 = build_sparse_tensor(requires_grad=False)
        self.context_cleanup_test_helper(rpc_args=(t1, t1), func=torch.add)

    @dist_init
    def test_context_cleanup_nested_rpc_sparse(self):
        t1 = build_sparse_tensor(requires_grad=True)
        t2 = build_sparse_tensor(requires_grad=True)
        dst_rank = (self.rank + 1) % self.world_size
        args = (t1, t2, dst_rank, self.world_size, 0)
        self.context_cleanup_test_helper(rpc_args=args, func=my_py_nested_call, nested=True)

    @dist_init
    def test_backward_no_grad_on_tensor_sparse(self):
        self._backward_no_grad_on_tensor(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), True)

    @dist_init
    def test_backward_simple_sparse(self):
        self._backward_simple(self._next_rank(), build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), None, True)

    @dist_init
    def test_backward_simple_self_sparse(self):
        self._backward_simple(self.rank, build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), None, True)

    @dist_init
    def test_backward_rref_multi_sparse(self):
        if self.rank > 0:
            callee = 'worker0'
            rref_owner = callee
            self._backward_rref(callee, rref_owner, build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), None, True)

    @dist_init
    def test_backward_rref_sparse(self):
        callee = worker_name(self._next_rank())
        rref_owner = callee
        self._backward_rref(callee, rref_owner, build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), None, True)

    @dist_init
    def test_backward_rref_nested_sparse(self):
        callee = worker_name((self.rank + 1) % self.world_size)
        rref_owner = worker_name((self.rank + 2) % self.world_size)
        self._backward_rref(callee, rref_owner, build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), None, True)

    @dist_init
    def test_trainer_ps_sparse(self):
        self._test_trainer_ps(build_sparse_tensor, _run_trainer, True)

    @dist_init
    def test_backward_multiple_round_trips_sparse(self):
        self._backward_multiple_round_trips(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=False), build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=False), build_sparse_tensor(requires_grad=True), None, True)

    @dist_init
    def test_backward_different_dtypes_sparse(self):
        self._backward_different_dtypes(build_sparse_tensor(requires_grad=True, dtype=torch.float32), build_sparse_tensor(requires_grad=True, dtype=torch.float64), True)

    @dist_init
    def test_backward_simple_python_udf_sparse(self):
        self._backward_simple_python_udf(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), True)

    @dist_init
    def test_backward_simple_script_call_sparse(self):
        self._backward_simple_script_call(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), True)

    @dist_init
    def test_nested_backward_accumulate_grads_sparse(self):
        self._nested_backward_accumulate_grads(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), True)

    @dist_init
    def test_backwards_nested_python_udf_sparse(self):
        self._backwards_nested_python_udf(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), True)

    @dist_init
    def test_mixed_requires_grad_sparse(self):
        self._mixed_requires_grad(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=False), True)

    @dist_init
    def test_multiple_backward_sparse(self):
        self._multiple_backward(build_sparse_tensor(requires_grad=True), build_sparse_tensor(requires_grad=True), True)

    @dist_init
    def test_embedding_bag_with_no_grad_tensors(self):
        dst = self._next_rank()
        remote_embedding = rpc.remote(worker_name(dst), torch.nn.EmbeddingBag, args=(16, 16), kwargs={'mode': 'sum', 'sparse': True})
        local_embedding = torch.nn.EmbeddingBag(16, 16, mode='sum', sparse=True)
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        per_sample_weights = torch.rand(8, requires_grad=True)
        offsets = torch.LongTensor([0, 4])
        local_res = local_embedding(input, offsets, per_sample_weights)
        torch.autograd.backward([local_res.sum()], retain_graph=True)
        torch.autograd.backward([local_res.sum()])
        local_grad = local_embedding.weight.grad
        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(worker_name(dst), DistAutogradTest._call_remote_embedding, args=(remote_embedding, input, offsets, per_sample_weights))
            dist_autograd.backward(context_id, [res.sum()], retain_graph=True)
            dist_autograd.backward(context_id, [res.sum()])
            remote_grad = rpc.rpc_sync(worker_name(dst), DistAutogradTest._get_grad, args=(remote_embedding, context_id))
            self.assertEqual(local_grad, remote_grad)