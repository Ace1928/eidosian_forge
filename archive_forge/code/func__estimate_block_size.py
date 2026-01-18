import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
def _estimate_block_size(self, min_run_time: float) -> int:
    with common.set_torch_threads(self._task_spec.num_threads):
        overhead = torch.tensor([self._timeit(0) for _ in range(5)]).median().item()
        number = 1
        while True:
            time_taken = self._timeit(number)
            relative_overhead = overhead / time_taken
            if relative_overhead <= 0.0001 and time_taken >= min_run_time / 1000:
                break
            if time_taken > min_run_time:
                break
            if number * 10 > 2147483647:
                break
            number *= 10
    return number