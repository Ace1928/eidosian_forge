import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def _parquet_byte_ranges(self, columns, row_groups=None, metadata=None, footer=None, footer_start=None):
    if metadata is not None:
        raise ValueError('metadata input not supported for PyarrowEngine')
    data_starts, data_ends = ([], [])
    md = self.pq.ParquetFile(io.BytesIO(footer)).metadata
    column_set = None if columns is None else set(columns)
    if column_set is not None:
        schema = md.schema.to_arrow_schema()
        has_pandas_metadata = schema.metadata is not None and b'pandas' in schema.metadata
        if has_pandas_metadata:
            md_index = [ind for ind in json.loads(schema.metadata[b'pandas'].decode('utf8')).get('index_columns', []) if not isinstance(ind, dict)]
            column_set |= set(md_index)
    for r in range(md.num_row_groups):
        if row_groups is None or r in row_groups:
            row_group = md.row_group(r)
            for c in range(row_group.num_columns):
                column = row_group.column(c)
                name = column.path_in_schema
                split_name = name.split('.')[0]
                if column_set is None or name in column_set or split_name in column_set:
                    file_offset0 = column.dictionary_page_offset
                    if file_offset0 is None:
                        file_offset0 = column.data_page_offset
                    num_bytes = column.total_compressed_size
                    if file_offset0 < footer_start:
                        data_starts.append(file_offset0)
                        data_ends.append(min(file_offset0 + num_bytes, footer_start))
    return (data_starts, data_ends)