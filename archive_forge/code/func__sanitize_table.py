from collections import defaultdict
from contextlib import nullcontext
from functools import reduce
import inspect
import json
import os
import re
import operator
import warnings
import pyarrow as pa
from pyarrow._parquet import (ParquetReader, Statistics,  # noqa
from pyarrow.fs import (LocalFileSystem, FileSystem, FileType,
from pyarrow import filesystem as legacyfs
from pyarrow.util import guid, _is_path_like, _stringify_path, _deprecate_api
def _sanitize_table(table, new_schema, flavor):
    if 'spark' in flavor:
        column_data = [table[i] for i in range(table.num_columns)]
        return pa.Table.from_arrays(column_data, schema=new_schema)
    else:
        return table