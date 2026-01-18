import _imp
import _io
import sys
import _warnings
import marshal
def _validate_timestamp_pyc(data, source_mtime, source_size, name, exc_details):
    """Validate a pyc against the source last-modified time.

    *data* is the contents of the pyc file. (Only the first 16 bytes are
    required.)

    *source_mtime* is the last modified timestamp of the source file.

    *source_size* is None or the size of the source file in bytes.

    *name* is the name of the module being imported. It is used for logging.

    *exc_details* is a dictionary passed to ImportError if it raised for
    improved debugging.

    An ImportError is raised if the bytecode is stale.

    """
    if _unpack_uint32(data[8:12]) != source_mtime & 4294967295:
        message = f'bytecode is stale for {name!r}'
        _bootstrap._verbose_message('{}', message)
        raise ImportError(message, **exc_details)
    if source_size is not None and _unpack_uint32(data[12:16]) != source_size & 4294967295:
        raise ImportError(f'bytecode is stale for {name!r}', **exc_details)