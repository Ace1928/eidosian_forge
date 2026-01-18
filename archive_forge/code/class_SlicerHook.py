from typing import TYPE_CHECKING, Callable, Union
import pandas
import ray
from modin.config import LazyExecution
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import disable_logging, get_logger
from modin.pandas.indexing import compute_sliced_len
from modin.utils import _inherit_docstrings
class SlicerHook(MaterializationHook):
    """
    Used by mask() for the slilced length computation.

    Parameters
    ----------
    ref : ObjectIDType
        Non-materialized length to be sliced.
    slc : slice
        The slice to be applied.
    """

    def __init__(self, ref: ObjectIDType, slc: slice):
        self.ref = ref
        self.slc = slc

    def pre_materialize(self):
        """
        Get the sliced length or object ref if not materialized.

        Returns
        -------
        int or ObjectIDType
        """
        if isinstance(self.ref, MetaListHook):
            len_or_ref = self.ref.pre_materialize()
            return compute_sliced_len(self.slc, len_or_ref) if isinstance(len_or_ref, int) else len_or_ref
        return self.ref

    def post_materialize(self, materialized):
        """
        Get the sliced length.

        Parameters
        ----------
        materialized : list or int

        Returns
        -------
        int
        """
        if isinstance(self.ref, MetaListHook):
            materialized = self.ref.post_materialize(materialized)
        return compute_sliced_len(self.slc, materialized)