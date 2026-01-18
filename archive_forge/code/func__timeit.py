import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
def _timeit(self, number: int) -> float:
    return max(self._timer.timeit(number), 1e-09)