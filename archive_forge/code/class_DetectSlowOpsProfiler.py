import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
class DetectSlowOpsProfiler(DispatcherWithoutBrokenFuncs):
    """
    Inspired from https://fb.workplace.com/groups/pytorch.dev/permalink/1054537595124720/
    """

    def __init__(self, main_profiler: _Profiler) -> None:
        self.main_profiler = main_profiler
        self.trace: List[_OpInfo] = []
        self.temp_disabled = False

    def _hardware_tflops_membw_limit(self, args: Tuple[Any, ...], outputs: Tuple[Any, ...]) -> Tuple[float, float]:
        device = None
        dtypes: List[torch.dtype] = []
        for a in itertools.chain(outputs, args):
            if isinstance(a, torch.Tensor):
                if device is None:
                    device = a.device
                dtypes.append(a.dtype)
        limits = get_device_limits(device)
        dtypes = [dt for dt in dtypes if dt in limits.gemm_tflops]
        if not dtypes or device is None:
            return (math.inf, math.inf)
        dtype = dtypes[0]
        if torch.is_autocast_enabled() and dtype is torch.float32:
            dtype = torch.get_autocast_gpu_dtype()
        return (limits.gemm_tflops[dtype], limits.gmem_bandwidth)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        func_packet = func._overloadpacket
        if self.temp_disabled or func_packet.__name__ in ['_record_function_exit', '_record_function_enter_new']:
            return func(*args, **kwargs)
        op = _OpInfo()
        op.ev_start.record()
        out = func(*args, **kwargs)
        op.ev_end.record()
        op.hardware_tflops_limit, op.hardware_membw_limit = self._hardware_tflops_membw_limit(args, out if isinstance(out, tuple) else (out,))
        op.op_name = func_packet.__name__
        self.temp_disabled = True
        flop_count = -1
        compute_flops = None
        if func_packet in FUNC_TO_XFORMERS_OPERATOR:
            flop_count = FUNC_TO_XFORMERS_OPERATOR[func_packet].operator_flop(*args, **kwargs)
        if flop_count == -1:
            compute_flops = flop_mapping.get(func_packet, guess_flops_unknown_op)
            flop_count = compute_flops(args, out if isinstance(out, tuple) else (out,))
            if isinstance(compute_flops, GemmOpComputeFlops):
                op.op_name += compute_flops.op_suffix(args)
        compute_io = io_mapping.get(func_packet, operation_memory_rw_bytes)
        op.io_bytes = compute_io(args, out if isinstance(out, tuple) else (out,))
        self.temp_disabled = False
        op.stacktrace = tuple(self.main_profiler.parents)
        op.flop_count = flop_count
        op.is_exact_flop = compute_flops is not guess_flops_unknown_op
        self.trace.append(op)
        return out

    def __enter__(self):
        self.main_profiler._install_hooks()
        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.main_profiler._remove_hooks()
        torch.cuda.synchronize()
        for op in self.trace:
            op.finalize()
        self.save_json()

    def step(self) -> None:
        pass

    def save_json(self) -> None:
        all_paths: Set[Tuple[str, ...]] = set()
        per_module_data: Dict[Tuple[str, ...], _OpInfoAggregated] = defaultdict(_OpInfoAggregated)
        per_op_data: Dict[str, _OpInfoAggregated] = defaultdict(_OpInfoAggregated)
        for op in self.trace:
            all_paths.add(op.stacktrace)
        for op in self.trace:
            for i in range(len(op.stacktrace)):
                if op.stacktrace[:i + 1] in all_paths:
                    per_module_data[op.stacktrace[:i + 1]].add(op)
            per_op_data[op.op_name].add(op)
        all_data = []
        for stacktrace, agg_info in per_module_data.items():
            all_data.append(agg_info.as_dict(agg='module', path=stacktrace, name=stacktrace[-1], op=''))
        for op_name, agg_info in per_op_data.items():
            paths_count: Dict[Tuple[str, ...], int] = defaultdict(int)
            agg_info.stacktraces.sort()
            for p in agg_info.stacktraces:
                paths_count[p] += 1
            maxp = agg_info.stacktraces[0]
            for p, count in paths_count.items():
                if count > paths_count[maxp]:
                    maxp = p
            all_data.append(agg_info.as_dict(agg='opname', path=f'{'.'.join(maxp)} (x{paths_count[maxp]})', name='', op=op_name))
        filename = self.main_profiler._create_output_filename('ops.json')
        self.main_profiler.summary.append(('OpsSummary', str(filename)))
        with open(filename, 'w+') as f:
            json.dump(all_data, f)