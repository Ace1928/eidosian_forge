import multiprocessing
import os
from typing import BinaryIO, Optional, Union
import fsspec
from .. import Dataset, Features, NamedSplit, config
from ..formatting import query_table
from ..packaged_modules.json.json import Json
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader
def _batch_json(self, args):
    offset, orient, lines, to_json_kwargs = args
    batch = query_table(table=self.dataset.data, key=slice(offset, offset + self.batch_size), indices=self.dataset._indices)
    json_str = batch.to_pandas().to_json(path_or_buf=None, orient=orient, lines=lines, **to_json_kwargs)
    if not json_str.endswith('\n'):
        json_str += '\n'
    return json_str.encode(self.encoding)