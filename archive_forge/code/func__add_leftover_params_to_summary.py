from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
@override
def _add_leftover_params_to_summary(self, arrays: List[Tuple[str, List[str]]], total_leftover_params: int) -> None:
    """Add summary of params not associated with module or layer to model summary."""
    super()._add_leftover_params_to_summary(arrays, total_leftover_params)
    layer_summaries = dict(arrays)
    layer_summaries['Params per Device'].append(NOT_APPLICABLE)