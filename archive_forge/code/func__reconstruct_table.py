from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _reconstruct_table(reduced_columns: List[Tuple[List['pyarrow.Array'], 'pyarrow.DataType']], schema: 'pyarrow.Schema') -> 'pyarrow.Table':
    """Restore a serialized Arrow Table, reconstructing each reduced column."""
    import pyarrow as pa
    columns = []
    for chunks_payload, type_ in reduced_columns:
        columns.append(_reconstruct_chunked_array(chunks_payload, type_))
    return pa.Table.from_arrays(columns, schema=schema)