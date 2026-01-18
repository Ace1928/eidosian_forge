import threading
from typing import Any, Dict
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
class KernelSideTable:
    id_to_kernel: Dict[int, Any] = dict()
    kernel_to_id: Dict[Any, int] = dict()
    lock = threading.Lock()

    def add_kernel(self, kernel) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]
            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    def get_kernel(self, idx: int):
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    def reset_table(self) -> None:
        self.id_to_kernel = dict()
        self.kernel_to_id = dict()