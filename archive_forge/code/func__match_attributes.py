from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _match_attributes(self, pn: Node, gn: Node) -> bool:
    assert isinstance(pn.target, str), f'pn.target {pn.target} must be a string.'
    assert isinstance(gn.target, str), f'gn.target {gn.target} must be a string.'
    pn_value = getattr(pn.graph.owning_module, pn.target)
    gn_value = getattr(gn.graph.owning_module, gn.target)
    if type(pn_value) != type(gn_value):
        return False
    if isinstance(pn_value, torch.Tensor):
        return isinstance(gn_value, torch.Tensor)
    else:
        raise RuntimeError(f'Unsupported type {pn_value} when matching attributes')
    return False