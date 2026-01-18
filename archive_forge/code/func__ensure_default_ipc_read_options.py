import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
def _ensure_default_ipc_read_options(options):
    if options and (not isinstance(options, IpcReadOptions)):
        raise TypeError('expected IpcReadOptions, got {}'.format(type(options)))
    return options or IpcReadOptions()