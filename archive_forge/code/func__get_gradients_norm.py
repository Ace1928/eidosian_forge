from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional
import torch
@torch.no_grad()
def _get_gradients_norm(self, params: List[torch.nn.Parameter]) -> float:
    grads = []
    for p in params:
        if p.grad is None:
            continue
        else:
            grads.append(p.grad.detach())
    if len(grads) == 0:
        return 0.0
    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, p=2, dtype=torch.float32) for g in grads]))
    return total_norm.item()