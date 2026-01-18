import pandas
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.utils import _inherit_docstrings
from .partition import PandasOnPythonDataframePartition
@_inherit_docstrings(PandasOnPythonDataframeAxisPartition.__init__)
class PandasOnPythonDataframeRowPartition(PandasOnPythonDataframeAxisPartition):
    axis = 1