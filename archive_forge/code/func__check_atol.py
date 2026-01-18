import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
def _check_atol(self, val_a: Union[float, torch.Tensor], val_b: Union[float, torch.Tensor]) -> bool:
    return self.atol is None or bool(abs(val_a - val_b) >= abs(self.atol))