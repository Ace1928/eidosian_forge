from typing import Dict, List, Optional
from torch import nn
def _get_module_by_path(module: nn.Module, path: str) -> nn.Module:
    path = path.split('.')
    for name in path:
        module = getattr(module, name)
    return module