import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from modin.config import AsyncReadMode
from modin.core.execution.modin_aqp import progress_bar_wrapper
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.generic.partitioning import (
from modin.logging import get_logger
from modin.utils import _inherit_docstrings
from .partition import PandasOnRayDataframePartition
from .virtual_partition import (
def _make_wrapped_method(name: str):
    """
    Define new attribute that should work with progress bar.

    Parameters
    ----------
    name : str
        Name of `GenericRayDataframePartitionManager` attribute that should be reused.

    Notes
    -----
    - `classmethod` decorator shouldn't be applied twice, so we refer to `__func__` attribute.
    - New attribute is defined for `PandasOnRayDataframePartitionManager`.
    """
    setattr(PandasOnRayDataframePartitionManager, name, classmethod(progress_bar_wrapper(getattr(GenericRayDataframePartitionManager, name).__func__)))