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
def _sanitize_schema(schema, flavor):
    if 'spark' in flavor:
        sanitized_fields = []
        schema_changed = False
        for field in schema:
            name = field.name
            sanitized_name = _sanitized_spark_field_name(name)
            if sanitized_name != name:
                schema_changed = True
                sanitized_field = pa.field(sanitized_name, field.type, field.nullable, field.metadata)
                sanitized_fields.append(sanitized_field)
            else:
                sanitized_fields.append(field)
        new_schema = pa.schema(sanitized_fields, metadata=schema.metadata)
        return (new_schema, schema_changed)
    else:
        return (schema, False)