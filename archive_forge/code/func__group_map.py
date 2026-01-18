from typing import Any, Callable, Dict, List, Optional, Type, Union
import pyarrow as pa
import ray
from duckdb import DuckDBPyConnection
from packaging import version
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.threading import RunOnce
from fugue import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue_duckdb.dataframe import DuckDataFrame
from fugue_duckdb.execution_engine import DuckExecutionEngine
from ._constants import FUGUE_RAY_DEFAULT_BATCH_SIZE, FUGUE_RAY_ZERO_COPY
from ._utils.cluster import get_default_partitions, get_default_shuffle_partitions
from ._utils.dataframe import add_coarse_partition_key, add_partition_key
from ._utils.io import RayIO
from .dataframe import RayDataFrame
def _group_map(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]=None) -> DataFrame:
    output_schema = Schema(output_schema)
    input_schema = df.schema
    presort = partition_spec.get_sorts(input_schema, with_partition_keys=partition_spec.algo == 'coarse')
    presort_tuples = [(k, 'ascending' if v else 'descending') for k, v in presort.items()]
    cursor = partition_spec.get_cursor(input_schema, 0)
    on_init_once: Any = None if on_init is None else RunOnce(on_init, lambda *args, **kwargs: to_uuid(id(on_init), id(args[0])))

    def _udf(adf: pa.Table) -> pa.Table:
        if adf.shape[0] == 0:
            return output_schema.create_empty_arrow_table()
        adf = adf.remove_column(len(input_schema))
        if len(partition_spec.presort) > 0:
            if version.parse(pa.__version__).major < 7:
                idx = pa.compute.sort_indices(adf, options=pa.compute.SortOptions(presort_tuples))
                adf = adf.take(idx)
            else:
                adf = adf.sort_by(presort_tuples)
        input_df = ArrowDataFrame(adf)
        if on_init_once is not None:
            on_init_once(0, input_df)
        cursor.set(lambda: input_df.peek_array(), 0, 0)
        output_df = map_func(cursor, input_df)
        return output_df.as_arrow()
    _df: RayDataFrame = self.execution_engine._to_ray_df(df)
    if partition_spec.num_partitions != '0':
        _df = self.execution_engine.repartition(_df, partition_spec)
    else:
        n = get_default_shuffle_partitions(self.execution_engine)
        if n > 0 and n != _df.num_partitions:
            _df = self.execution_engine.repartition(_df, PartitionSpec(num=n))
    if partition_spec.algo != 'coarse':
        rdf, _ = add_partition_key(_df.native, keys=partition_spec.partition_by, input_schema=input_schema, output_key=_RAY_PARTITION_KEY)
    else:
        rdf = add_coarse_partition_key(_df.native, keys=partition_spec.partition_by, output_key=_RAY_PARTITION_KEY, bucket=_df.num_partitions)
    gdf = rdf.groupby(_RAY_PARTITION_KEY)
    sdf = gdf.map_groups(_udf, batch_format='pyarrow', **self.execution_engine._get_remote_args())
    return RayDataFrame(sdf, schema=output_schema, internal_schema=True)