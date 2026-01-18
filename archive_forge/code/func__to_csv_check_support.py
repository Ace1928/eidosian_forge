import io
import pandas
from pandas.io.common import get_handle, stringify_path
from ray.data import from_pandas_refs
from modin.core.execution.ray.common import RayWrapper, SignalActor
from modin.core.execution.ray.generic.io import RayIO
from modin.core.io import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.distributed.dataframe.pandas.partitions import (
from modin.experimental.core.io import (
from modin.experimental.core.storage_formats.pandas.parsers import (
from ..dataframe import PandasOnRayDataframe
from ..partitioning import PandasOnRayDataframePartition
@staticmethod
def _to_csv_check_support(kwargs):
    """
        Check if parallel version of ``to_csv`` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``.to_csv()``.

        Returns
        -------
        bool
            Whether parallel version of ``to_csv`` is applicable.
        """
    path_or_buf = kwargs['path_or_buf']
    compression = kwargs['compression']
    if not isinstance(path_or_buf, str):
        return False
    if 'r' in kwargs['mode'] and '+' in kwargs['mode']:
        return False
    if kwargs['encoding'] is not None:
        encoding = kwargs['encoding'].lower()
        if 'u' in encoding or 'utf' in encoding:
            if '16' in encoding or '32' in encoding:
                return False
    if compression is None or not compression == 'infer':
        return False
    if any((path_or_buf.endswith(ext) for ext in ['.gz', '.bz2', '.zip', '.xz'])):
        return False
    return True