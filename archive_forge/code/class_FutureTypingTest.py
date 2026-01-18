import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
class FutureTypingTest:

    @dist_init
    def test_future_passed_between_python_and_jit(self):
        dst_rank = (self.rank + 1) % self.world_size
        inputs = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        ret_fut = rpc.rpc_async(worker_name(dst_rank), two_args_two_kwargs, args=inputs)
        expected_res = torch.tensor([10, 10])

        @torch.jit.script
        def future_wait_in_script(fut: Future[Tensor]) -> Tensor:
            return fut.wait()
        self.assertEqual(future_wait_in_script(ret_fut), expected_res)

        @torch.jit.script
        def future_return_to_python(dst_rank: int, inputs: Tuple[Tensor, Tensor]) -> Future[Tensor]:
            return rpc.rpc_async(f'worker{dst_rank}', two_args_two_kwargs, inputs)
        fut_res = future_return_to_python(dst_rank, inputs)
        self.assertEqual(fut_res.wait(), expected_res)

    @dist_init
    def test_future_python_annotation(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        input_0 = torch.ones(2, 2)
        input_1 = 1
        expected_res = torch.add(input_0, input_1)

        @torch.jit.ignore
        def python_return_future() -> Future[Tensor]:
            fut = rpc.rpc_async(dst_worker_name, torch.add, (input_0, input_1), {})
            return fut

        @torch.jit.script
        def script_use_future() -> Tensor:
            fut = python_return_future()
            return fut.wait()
        res = script_use_future()
        self.assertEqual(res, expected_res)