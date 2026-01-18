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
class PandasOnDaskIO(BaseIO):
    """The class implements interface in ``BaseIO`` using Dask as an execution engine."""
    frame_cls = PandasOnDaskDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(frame_cls=PandasOnDaskDataframe, frame_partition_cls=PandasOnDaskDataframePartition, query_compiler_cls=PandasQueryCompiler, base_io=BaseIO)

    def __make_read(*classes, build_args=build_args):
        return type('', (DaskWrapper, *classes), build_args).read

    def __make_write(*classes, build_args=build_args):
        return type('', (DaskWrapper, *classes), build_args).write
    read_csv = __make_read(PandasCSVParser, CSVDispatcher)
    read_fwf = __make_read(PandasFWFParser, FWFDispatcher)
    read_json = __make_read(PandasJSONParser, JSONDispatcher)
    read_parquet = __make_read(PandasParquetParser, ParquetDispatcher)
    to_parquet = __make_write(ParquetDispatcher)
    read_feather = __make_read(PandasFeatherParser, FeatherDispatcher)
    read_sql = __make_read(PandasSQLParser, SQLDispatcher)
    to_sql = __make_write(SQLDispatcher)
    read_excel = __make_read(PandasExcelParser, ExcelDispatcher)
    read_csv_glob = __make_read(ExperimentalPandasCSVGlobParser, ExperimentalCSVGlobDispatcher)
    read_parquet_glob = __make_read(ExperimentalPandasParquetParser, ExperimentalGlobDispatcher)
    to_parquet_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': BaseIO.to_parquet})
    read_json_glob = __make_read(ExperimentalPandasJsonParser, ExperimentalGlobDispatcher)
    to_json_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': BaseIO.to_json})
    read_xml_glob = __make_read(ExperimentalPandasXmlParser, ExperimentalGlobDispatcher)
    to_xml_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': BaseIO.to_xml})
    read_pickle_glob = __make_read(ExperimentalPandasPickleParser, ExperimentalGlobDispatcher)
    to_pickle_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': BaseIO.to_pickle})
    read_custom_text = __make_read(ExperimentalCustomTextParser, ExperimentalCustomTextDispatcher)
    read_sql_distributed = __make_read(ExperimentalSQLDispatcher, build_args={**build_args, 'base_read': read_sql})
    del __make_read
    del __make_write

    @classmethod
    def from_dask(cls, dask_obj):
        """
        Create a Modin `query_compiler` from a Dask DataFrame.

        Parameters
        ----------
        dask_obj : dask.dataframe.DataFrame
            The Dask DataFrame to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Dask DataFrame.
        """
        client = default_client()
        dask_fututures = client.compute(dask_obj.to_delayed())
        modin_df = from_partitions(dask_fututures, axis=0)._query_compiler
        return modin_df

    @classmethod
    def to_dask(cls, modin_obj):
        """
        Convert a Modin DataFrame/Series to a Dask DataFrame/Series.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to convert.

        Returns
        -------
        dask.dataframe.DataFrame or dask.dataframe.Series
            Converted object with type depending on input.
        """
        from dask.dataframe import from_delayed
        partitions = unwrap_partitions(modin_obj, axis=0)
        if isinstance(modin_obj, Series):
            client = default_client()

            def df_to_series(df):
                series = df[df.columns[0]]
                if df.columns[0] == MODIN_UNNAMED_SERIES_LABEL:
                    series.name = None
                return series
            partitions = [client.submit(df_to_series, part) for part in partitions]
        return from_delayed(partitions)