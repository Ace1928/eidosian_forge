from typing import Dict, List, Optional
from torch import nn
def _set_module_by_path(module: nn.Module, path: str, value: nn.Module) -> None:
    path = path.split('.')
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)