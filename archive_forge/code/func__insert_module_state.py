from typing import cast, Dict, Optional
import torch.nn as nn
def _insert_module_state(module: nn.Module, state: _State) -> None:
    global _module_state_mapping
    assert module not in _module_state_mapping, f'Inserting {module} more than once.'
    _module_state_mapping[module] = state