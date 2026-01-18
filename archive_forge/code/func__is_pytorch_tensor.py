import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
def _is_pytorch_tensor(self, tensor: NumpyArrayOrPyTorchTensor) -> bool:
    return hasattr(tensor, '__module__') and tensor.__module__ == 'torch'