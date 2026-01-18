from distributed.client import default_client
from modin.core.execution.dask.common import DaskWrapper
from modin.core.execution.dask.implementations.pandas_on_dask.dataframe import (
from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
from modin.core.io import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.distributed.dataframe.pandas.partitions import (
from modin.experimental.core.io import (
from modin.experimental.core.storage_formats.pandas.parsers import (
from modin.pandas.series import Series
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def __make_write(*classes, build_args=build_args):
    return type('', (DaskWrapper, *classes), build_args).write