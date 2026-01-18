import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def _make_dataset_for_pickling(tempdir, N=100):
    path = tempdir / 'data.parquet'
    fs = LocalFileSystem._get_instance()
    df = pd.DataFrame({'index': np.arange(N), 'values': np.random.randn(N)}, columns=['index', 'values'])
    table = pa.Table.from_pandas(df)
    num_groups = 3
    with pq.ParquetWriter(path, table.schema) as writer:
        for i in range(num_groups):
            writer.write_table(table)
    reader = pq.ParquetFile(path)
    assert reader.metadata.num_row_groups == num_groups
    metadata_path = tempdir / '_metadata'
    with fs.open(metadata_path, 'wb') as f:
        pq.write_metadata(table.schema, f)
    dataset = pq.ParquetDataset(tempdir, filesystem=fs)
    return dataset