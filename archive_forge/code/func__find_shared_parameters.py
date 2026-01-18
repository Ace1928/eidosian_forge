from typing import Dict, List, Optional
from torch import nn
def _find_shared_parameters(module: nn.Module, tied_parameters: Optional[Dict]=None, prefix: str='') -> List[str]:
    if tied_parameters is None:
        tied_parameters = {}
    for name, param in module._parameters.items():
        param_prefix = prefix + ('.' if prefix else '') + name
        if param is None:
            continue
        if param not in tied_parameters:
            tied_parameters[param] = []
        tied_parameters[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        _find_shared_parameters(m, tied_parameters, submodule_prefix)
    return [x for x in tied_parameters.values() if len(x) > 1]