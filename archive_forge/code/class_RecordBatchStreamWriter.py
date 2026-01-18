import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
class RecordBatchStreamWriter(lib._RecordBatchStreamWriter):
    __doc__ = 'Writer for the Arrow streaming binary format\n\n{}'.format(_ipc_writer_class_doc)

    def __init__(self, sink, schema, *, use_legacy_format=None, options=None):
        options = _get_legacy_format_default(use_legacy_format, options)
        self._open(sink, schema, options=options)