import multiprocessing
import os
import sys
from functools import partial
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, no_type_check
from unittest.mock import Mock
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import select_topk, to_onehot
from torchmetrics.utilities.enums import DataType
def _try_proceed_with_timeout(fn: Callable, timeout: int=_DOCTEST_DOWNLOAD_TIMEOUT) -> bool:
    """Check if a certain function is taking too long to execute.

    Function will only be executed if running inside a doctest context. Currently, does not support Windows.

    Args:
        fn: function to check
        timeout: timeout for function

    Returns:
        Bool indicating if the function finished within the specified timeout

    """
    proc = multiprocessing.Process(target=fn)
    print(f'Trying to run `{fn.__name__}` for {timeout}s...', file=sys.stderr)
    proc.start()
    proc.join(timeout)
    if not proc.is_alive():
        return True
    print(f'`{fn.__name__}` did not complete with {timeout}, killing process and returning False', file=sys.stderr)
    proc.kill()
    return False