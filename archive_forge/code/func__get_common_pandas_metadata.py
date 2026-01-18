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
def _get_common_pandas_metadata(self):
    if not self._base_dir:
        return None
    metadata = None
    for name in ['_common_metadata', '_metadata']:
        metadata_path = os.path.join(str(self._base_dir), name)
        finfo = self.filesystem.get_file_info(metadata_path)
        if finfo.is_file:
            pq_meta = read_metadata(metadata_path, filesystem=self.filesystem)
            metadata = pq_meta.metadata
            if metadata and b'pandas' in metadata:
                break
    return metadata