import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class InfNanRemoveLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.

    This logits processor has no `generate` example, as there shouldn't be a correct combination of flags that warrants
    its use.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[scores != scores] = 0.0
        scores[scores == float('inf')] = torch.finfo(scores.dtype).max
        scores[scores == float('-inf')] = torch.finfo(scores.dtype).min
        return scores