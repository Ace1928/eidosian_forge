import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _ensure_format(obj):
    if isinstance(obj, FileFormat):
        return obj
    elif obj == 'parquet':
        if not _parquet_available:
            raise ValueError(_parquet_msg)
        return ParquetFileFormat()
    elif obj in {'ipc', 'arrow'}:
        return IpcFileFormat()
    elif obj == 'feather':
        return FeatherFileFormat()
    elif obj == 'csv':
        return CsvFileFormat()
    elif obj == 'orc':
        if not _orc_available:
            raise ValueError(_orc_msg)
        return OrcFileFormat()
    elif obj == 'json':
        return JsonFileFormat()
    else:
        raise ValueError("format '{}' is not supported".format(obj))