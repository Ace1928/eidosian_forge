from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _to_calcite(self, op):
    """
        Translate tree to a calcite node sequence.

        Parameters
        ----------
        op : DFAlgNode
            A tree to translate.

        Returns
        -------
        CalciteBaseNode
            The last node of the generated sequence.
        """
    with self._set_input_ctx(op):
        if isinstance(op, FrameNode):
            self._process_frame(op)
        elif isinstance(op, MaskNode):
            self._process_mask(op)
        elif isinstance(op, GroupbyAggNode):
            self._process_groupby(op)
        elif isinstance(op, TransformNode):
            self._process_transform(op)
        elif isinstance(op, JoinNode):
            self._process_join(op)
        elif isinstance(op, UnionNode):
            self._process_union(op)
        elif isinstance(op, SortNode):
            self._process_sort(op)
        elif isinstance(op, FilterNode):
            self._process_filter(op)
        else:
            raise NotImplementedError(f"CalciteBuilder doesn't support {type(op).__name__}")
    return self.res[-1]