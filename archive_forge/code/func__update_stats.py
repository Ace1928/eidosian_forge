import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
def _update_stats(self, val: torch.Tensor) -> None:
    self.running_mean.update(val)
    self.last_val = val