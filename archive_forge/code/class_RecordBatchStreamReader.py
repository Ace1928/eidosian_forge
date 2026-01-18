import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
class RecordBatchStreamReader(lib._RecordBatchStreamReader):
    """
    Reader for the Arrow streaming binary format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
        If you want to use memory map use MemoryMappedFile as source.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC deserialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.
    """

    def __init__(self, source, *, options=None, memory_pool=None):
        options = _ensure_default_ipc_read_options(options)
        self._open(source, options=options, memory_pool=memory_pool)