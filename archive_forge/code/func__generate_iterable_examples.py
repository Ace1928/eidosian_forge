import os
import posixpath
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
import datasets
from datasets.arrow_writer import ArrowWriter, ParquetWriter
from datasets.config import MAX_SHARD_SIZE
from datasets.filesystems import (
from datasets.iterable_dataset import _BaseExamplesIterable
from datasets.utils.py_utils import convert_file_size_to_int
def _generate_iterable_examples(df: 'pyspark.sql.DataFrame', partition_order: List[int]):
    import pyspark

    def generate_fn():
        df_with_partition_id = df.select('*', pyspark.sql.functions.spark_partition_id().alias('part_id'))
        partition_df = _reorder_dataframe_by_partition(df_with_partition_id, partition_order)
        row_id = 0
        rows = partition_df.toLocalIterator(prefetchPartitions=True)
        curr_partition = -1
        for row in rows:
            row_as_dict = row.asDict()
            part_id = row_as_dict['part_id']
            row_as_dict.pop('part_id')
            if curr_partition != part_id:
                curr_partition = part_id
                row_id = 0
            yield (f'{part_id}_{row_id}', row_as_dict)
            row_id += 1
    return generate_fn