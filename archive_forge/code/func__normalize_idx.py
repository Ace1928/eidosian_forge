import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from ... import ir
from ...autotune_process import CUDABenchmarkRequest
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V
from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP
def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length