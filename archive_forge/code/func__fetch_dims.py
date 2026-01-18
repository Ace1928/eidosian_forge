import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
def _fetch_dims(tree: Union[dict, list, tuple, torch.Tensor]) -> List[Tuple[int, ...]]:
    shapes = []
    if isinstance(tree, dict):
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif isinstance(tree, (list, tuple)):
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif isinstance(tree, torch.Tensor):
        shapes.append(tree.shape)
    else:
        raise ValueError('Not supported')
    return shapes