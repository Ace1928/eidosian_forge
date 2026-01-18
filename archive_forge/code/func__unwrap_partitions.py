from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from pandas._typing import Axes
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.pandas.dataframe import DataFrame, Series
def _unwrap_partitions() -> list:
    [p.drain_call_queue() for p in modin_frame._partitions.flatten()]

    def get_block(partition: PartitionUnionType) -> np.ndarray:
        if hasattr(partition, 'force_materialization'):
            blocks = partition.force_materialization().list_of_blocks
        else:
            blocks = partition.list_of_blocks
        assert len(blocks) == 1, f'Implementation assumes that partition contains a single block, but {len(blocks)} recieved.'
        return blocks[0]
    if get_ip:
        return [[(partition.ip(materialize=False), get_block(partition)) for partition in row] for row in modin_frame._partitions]
    else:
        return [[get_block(partition) for partition in row] for row in modin_frame._partitions]