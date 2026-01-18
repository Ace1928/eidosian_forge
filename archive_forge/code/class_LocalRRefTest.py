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
class LocalRRefTest:

    @dist_init
    def test_create_local_script_class_rref_in_py(self):
        if self.rank != 0:
            return
        rref_script_class = rpc.RRef(MyScriptClass(self.rank))
        ret = rref_script_class.to_here().get_value()
        self.assertEqual(ret, self.rank)

    @dist_init
    def test_create_local_script_module_rref_in_py(self):
        if self.rank != 0:
            return
        rref_script_module = rpc.RRef(MyScriptModule(self.rank), MyModuleInterface)
        ret = rref_script_module.to_here().forward()
        self.assertEqual(ret, torch.ones(self.rank))
        with self.assertRaisesRegex(RuntimeError, 'The RRef being created contains a ScriptModule, must provide its ModuleInterface type hint.'):
            rref_script_module = rpc.RRef(MyScriptModule(self.rank))

    @dist_init
    def test_return_local_script_class_rref_in_py_and_use_in_script(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.rpc_sync(dst_worker_name, owner_create_rref_my_script_class, args=(self.rank,))

        def use_rref_on_owner(rref: RRef[MyScriptClass]) -> int:
            args = (rref,)
            kwargs: Dict[str, Any] = {}
            fut = rpc.rpc_async(rref.owner(), script_rref_get_value_my_script_class, args, kwargs)
            ret = fut.wait()
            return ret
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, self.rank)
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, self.rank)

    @dist_init
    def test_return_local_script_module_rref_in_py_and_use_in_script(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.rpc_sync(dst_worker_name, owner_create_rref_my_script_module, args=(self.rank,))

        def use_rref_on_owner(rref: RRef[MyModuleInterface]) -> Tensor:
            args = (rref,)
            kwargs: Dict[str, Any] = {}
            fut = rpc.rpc_async(rref.owner_name(), script_rref_run_forward_my_script_module, args, kwargs)
            ret = fut.wait()
            return ret
        ret = use_rref_on_owner(rref)
        self.assertEqual(ret, torch.ones(self.rank))
        use_rref_on_owner_script = torch.jit.script(use_rref_on_owner)
        ret = use_rref_on_owner_script(rref)
        self.assertEqual(ret, torch.ones(self.rank))