from __future__ import annotations
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _meta_from_dtypes
@classmethod
def _aggregate_files(cls, aggregate_files, split_stripes, parts):
    if aggregate_files is True and int(split_stripes) > 1 and (len(parts) > 1):
        new_parts = []
        new_part = parts[0]
        nstripes = len(new_part[0][1])
        for part in parts[1:]:
            next_nstripes = len(part[0][1])
            if next_nstripes + nstripes <= split_stripes:
                new_part.append(part[0])
                nstripes += next_nstripes
            else:
                new_parts.append(new_part)
                new_part = part
                nstripes = next_nstripes
        new_parts.append(new_part)
        return new_parts
    else:
        return parts