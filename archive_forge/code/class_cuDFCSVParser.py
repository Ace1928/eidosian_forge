import warnings
from io import BytesIO
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from modin.config import MinPartitionSize
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import (
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.error_message import ErrorMessage
class cuDFCSVParser(cuDFParser):

    @classmethod
    def parse(cls, fname, **kwargs):
        warnings.filterwarnings('ignore')
        num_splits = kwargs.pop('num_splits', None)
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        index_col = kwargs.get('index_col', None)
        gpu_selected = kwargs.pop('gpu', 0)
        if start is not None and end is not None:
            put_func = cls.frame_partition_cls.put
            with OpenFile(fname, 'rb', kwargs.pop('compression', 'infer')) as bio:
                if kwargs.get('encoding', None) is not None:
                    header = b'' + bio.readline()
                else:
                    header = b''
                bio.seek(start)
                to_read = header + bio.read(end - start)
            pandas_df = pandas.read_csv(BytesIO(to_read), **kwargs)
        else:
            pandas_df = pandas.read_csv(fname, **kwargs)
            num_splits = 1
        if index_col is not None:
            index = pandas_df.index
        else:
            index = len(pandas_df)
        partition_dfs = _split_result_for_readers(1, num_splits, pandas_df)
        key = [put_func(GPU_MANAGERS[gpu_selected], partition_df) for partition_df in partition_dfs]
        return key + [index, pandas_df.dtypes]