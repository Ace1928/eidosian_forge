import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
def _is_better(self, diff_val: torch.Tensor) -> bool:
    if self.mode == 'min':
        return bool((diff_val <= 0.0).all())
    if self.mode == 'max':
        return bool((diff_val >= 0).all())
    raise ValueError(f'Invalid mode. Has to be min or max, found {self.mode}')