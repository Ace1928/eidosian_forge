import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def _set_engine(engine_str):
    if engine_str == 'auto':
        try_engines = ('fastparquet', 'pyarrow')
    elif not isinstance(engine_str, str):
        raise ValueError("Failed to set parquet engine! Please pass 'fastparquet', 'pyarrow', or 'auto'")
    elif engine_str not in ('fastparquet', 'pyarrow'):
        raise ValueError(f'{engine_str} engine not supported by `fsspec.parquet`')
    else:
        try_engines = [engine_str]
    for engine in try_engines:
        try:
            if engine == 'fastparquet':
                return FastparquetEngine()
            elif engine == 'pyarrow':
                return PyarrowEngine()
        except ImportError:
            pass
    raise ImportError(f"The following parquet engines are not installed in your python environment: {try_engines}.Please install 'fastparquert' or 'pyarrow' to utilize the `fsspec.parquet` module.")