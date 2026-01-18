import pandas
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.utils import _inherit_docstrings
from .partition import PandasOnPythonDataframePartition
class PandasOnPythonDataframeAxisPartition(PandasDataframeAxisPartition):
    """
    Class defines axis partition interface with pandas storage format and Python engine.

    Inherits functionality from ``PandasDataframeAxisPartition`` class.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnPythonDataframePartition]
        List of ``PandasOnPythonDataframePartition`` and
        ``PandasOnPythonDataframeVirtualPartition`` objects, or a single
        ``PandasOnPythonDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """
    partition_type = PandasOnPythonDataframePartition
    instance_type = pandas.DataFrame