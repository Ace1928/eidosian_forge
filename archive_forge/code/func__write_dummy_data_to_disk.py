import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def _write_dummy_data_to_disk(tmpdir, file_name, table):
    path = os.path.join(str(tmpdir), file_name)
    with pa.ipc.RecordBatchFileWriter(path, schema=table.schema) as writer:
        writer.write_table(table)
    return path