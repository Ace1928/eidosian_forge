import multiprocessing
import os
from typing import BinaryIO, Optional, Union
from .. import Dataset, Features, NamedSplit, config
from ..formatting import query_table
from ..packaged_modules.csv.csv import Csv
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader
def _batch_csv(self, args):
    offset, header, index, to_csv_kwargs = args
    batch = query_table(table=self.dataset.data, key=slice(offset, offset + self.batch_size), indices=self.dataset._indices)
    csv_str = batch.to_pandas().to_csv(path_or_buf=None, header=header if offset == 0 else False, index=index, **to_csv_kwargs)
    return csv_str.encode(self.encoding)