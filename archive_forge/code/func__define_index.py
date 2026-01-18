import codecs
import io
import os
import warnings
from csv import QUOTE_NONE
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.io.text.utils import CustomNewlineIterator
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
@classmethod
def _define_index(cls, index_ids: list, index_name: str) -> Tuple[IndexColType, list]:
    """
        Compute the resulting DataFrame index and index lengths for each of partitions.

        Parameters
        ----------
        index_ids : list
            Array with references to the partitions index objects.
        index_name : str
            Name that should be assigned to the index if `index_col`
            is not provided.

        Returns
        -------
        new_index : IndexColType
            Index that should be passed to the new_frame constructor.
        row_lengths : list
            Partitions rows lengths.
        """
    index_objs = cls.materialize(index_ids)
    if len(index_objs) == 0 or isinstance(index_objs[0], int):
        row_lengths = index_objs
        new_index = pandas.RangeIndex(sum(index_objs))
    else:
        row_lengths = [len(o) for o in index_objs]
        new_index = index_objs[0].append(index_objs[1:])
        new_index.name = index_name
    return (new_index, row_lengths)