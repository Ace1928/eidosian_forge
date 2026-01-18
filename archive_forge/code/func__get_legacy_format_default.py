import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
def _get_legacy_format_default(use_legacy_format, options):
    if use_legacy_format is not None and options is not None:
        raise ValueError('Can provide at most one of options and use_legacy_format')
    elif options:
        if not isinstance(options, IpcWriteOptions):
            raise TypeError('expected IpcWriteOptions, got {}'.format(type(options)))
        return options
    metadata_version = MetadataVersion.V5
    if use_legacy_format is None:
        use_legacy_format = bool(int(os.environ.get('ARROW_PRE_0_15_IPC_FORMAT', '0')))
    if bool(int(os.environ.get('ARROW_PRE_1_0_METADATA_VERSION', '0'))):
        metadata_version = MetadataVersion.V4
    return IpcWriteOptions(use_legacy_format=use_legacy_format, metadata_version=metadata_version)