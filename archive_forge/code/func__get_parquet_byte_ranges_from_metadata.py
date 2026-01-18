import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def _get_parquet_byte_ranges_from_metadata(metadata, fs, engine, columns=None, row_groups=None, max_gap=64000, max_block=256000000):
    """Simplified version of `_get_parquet_byte_ranges` for
    the case that an engine-specific `metadata` object is
    provided, and the remote footer metadata does not need to
    be transferred before calculating the required byte ranges.
    """
    data_paths, data_starts, data_ends = engine._parquet_byte_ranges(columns, row_groups=row_groups, metadata=metadata)
    data_paths, data_starts, data_ends = merge_offset_ranges(data_paths, data_starts, data_ends, max_gap=max_gap, max_block=max_block, sort=False)
    result = {fn: {} for fn in list(set(data_paths))}
    _transfer_ranges(fs, result, data_paths, data_starts, data_ends)
    _add_header_magic(result)
    return result