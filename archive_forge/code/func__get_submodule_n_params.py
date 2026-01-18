from typing import Any, List, Tuple
import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import (
def _get_submodule_n_params(module: nn.Module, path: str):
    """
    Get submodule and the direct path of parameter from the module
    """
    if '.' in path:
        path_list = path.split('.')
        parent_module_path = '.'.join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return (module, path)