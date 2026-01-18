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
class RRefAPITest:

    @dist_init
    def test_rref_is_owner(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        rref_var = rpc_return_rref(dst_worker_name)

        @torch.jit.script
        def rref_tensor_is_owner(rref_var: RRef[Tensor]) -> bool:
            return rref_var.is_owner()
        res = rref_tensor_is_owner(rref_var)
        self.assertEqual(res, False)

    @dist_init
    def test_rref_local_value(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        rref = rpc_return_rref(dst_worker_name)
        with self.assertRaisesRegex(RuntimeError, "Can't call RRef.local_value\\(\\) on a non-owner RRef"):
            rref_local_value(rref)
        ret = ret = rpc.rpc_sync(dst_worker_name, rref_local_value, (rref,))
        self.assertEqual(ret, torch.add(torch.ones(2, 2), 1))

    @dist_init
    def test_local_rref_local_value(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name(self.rank)
        rref = rpc.remote(dst_worker_name, return_value, (5,), {})
        ret = rref_local_value(rref)
        self.assertEqual(ret, 5)

    def _create_rref(self):
        owner_rank = (self.rank + 2) % self.world_size
        return rpc.remote(worker_name(owner_rank), torch.add, args=(torch.zeros(2, 2), 1))

    @dist_init
    def test_user_rrefs_confirmed(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret = rpc.rpc_sync(worker_name(dst_rank), script_check_rref_confirmed, args=(rref,))
        self.assertEqual(ret, True)

    @dist_init
    def test_user_rrefs_confirmed_remote(self):
        dst_rank = (self.rank + 1) % self.world_size
        rref = self._create_rref()
        ret_rref = rpc.remote(worker_name(dst_rank), script_check_rref_confirmed, args=(rref,))
        self.assertEqual(ret_rref.to_here(), True)

    @dist_init
    def test_rref_list_mutate(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        list_rref = rpc.remote(dst, list_create)
        rpc.rpc_sync(dst, rref_list_mutate, args=(list_rref,))
        self.assertEqual(list_rref.to_here(), [1, 2, 3, 4, 5, 6])