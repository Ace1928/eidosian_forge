from typing import Literal, Optional, Sequence, Union
import torch
from torch import Tensor
from torchmetrics.functional.text.helper import _LevenshteinEditDistance as _LE_distance
def _edit_distance_update(preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]], substitution_cost: int=1) -> Tensor:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    if not all((isinstance(x, str) for x in preds)):
        raise ValueError(f'Expected all values in argument `preds` to be string type, but got {preds}')
    if not all((isinstance(x, str) for x in target)):
        raise ValueError(f'Expected all values in argument `target` to be string type, but got {target}')
    if len(preds) != len(target):
        raise ValueError(f'Expected argument `preds` and `target` to have same length, but got {len(preds)} and {len(target)}')
    distance = [_LE_distance(t, op_substitute=substitution_cost)(p)[0] for p, t in zip(preds, target)]
    return torch.tensor(distance, dtype=torch.int)