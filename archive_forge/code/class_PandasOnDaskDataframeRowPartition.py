import pandas
from distributed import Future
from distributed.utils import get_ip
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.dask.common import DaskWrapper
from modin.utils import _inherit_docstrings
from .partition import PandasOnDaskDataframePartition
@_inherit_docstrings(PandasOnDaskDataframeVirtualPartition.__init__)
class PandasOnDaskDataframeRowPartition(PandasOnDaskDataframeVirtualPartition):
    axis = 1