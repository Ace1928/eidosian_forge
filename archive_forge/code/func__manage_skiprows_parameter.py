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
def _manage_skiprows_parameter(cls, skiprows: Union[int, Sequence[int], Callable, None]=None, header_size: int=0) -> Tuple[Union[int, Sequence, Callable], bool, int]:
    """
        Manage `skiprows` parameter of read_csv and read_fwf functions.

        Change `skiprows` parameter in the way Modin could more optimally
        process it. `csv_dispatcher` and `fwf_dispatcher` have two mechanisms of rows skipping:

        1) During file partitioning (setting of file limits that should be read
        by each partition) exact rows can be excluded from partitioning scope,
        thus they won't be read at all and can be considered as skipped. This is
        the most effective way of rows skipping (since it doesn't require any
        actual data reading and postprocessing), but in this case `skiprows`
        parameter can be an integer only. When it possible Modin always uses
        this approach by setting of `skiprows_partitioning` return value.

        2) Rows for skipping can be dropped after full dataset import. This is
        more expensive way since it requires extra IO work and postprocessing
        afterwards, but `skiprows` parameter can be of any non-integer type
        supported by any pandas read function. These rows is
        specified by setting of `skiprows_md` return value.

        In some cases, if `skiprows` is uniformly distributed array (e.g. [1,2,3]),
        `skiprows` can be "squashed" and represented as integer to make a fastpath.
        If there is a gap between the first row for skipping and the last line of
        the header (that will be skipped too), then assign to read this gap first
        (assign the first partition to read these rows be setting of `pre_reading`
        return value). See `Examples` section for details.

        Parameters
        ----------
        skiprows : int, array or callable, optional
            Original `skiprows` parameter of any pandas read function.
        header_size : int, default: 0
            Number of rows that are used by header.

        Returns
        -------
        skiprows_md : int, array or callable
            Updated skiprows parameter. If `skiprows` is an array, this
            array will be sorted. Also parameter will be aligned to
            actual data in the `query_compiler` (which, for example,
            doesn't contain header rows)
        pre_reading : int
            The number of rows that should be read before data file
            splitting for further reading (the number of rows for
            the first partition).
        skiprows_partitioning : int
            The number of rows that should be skipped virtually (skipped during
            data file partitioning).

        Examples
        --------
        Let's consider case when `header`="infer" and `skiprows`=[3,4,5]. In
        this specific case fastpath can be done since `skiprows` is uniformly
        distributed array, so we can "squash" it to integer and set
        `skiprows_partitioning`=3. But if no additional action will be done,
        these three rows will be skipped right after header line, that corresponds
        to `skiprows`=[1,2,3]. Now, to avoid this discrepancy, we need to assign
        the first partition to read data between header line and the first
        row for skipping by setting of `pre_reading` parameter, so setting
        `pre_reading`=2. During data file partitiong, these lines will be assigned
        for reading for the first partition, and then file position will be set at
        the beginning of rows that should be skipped by `skiprows_partitioning`.
        After skipping of these rows, the rest data will be divided between the
        rest of partitions, see rows assignement below:

        0 - header line (skip during partitioning)
        1 - pre_reading (assign to read by the first partition)
        2 - pre_reading (assign to read by the first partition)
        3 - skiprows_partitioning (skip during partitioning)
        4 - skiprows_partitioning (skip during partitioning)
        5 - skiprows_partitioning (skip during partitioning)
        6 - data to partition (divide between the rest of partitions)
        7 - data to partition (divide between the rest of partitions)
        """
    pre_reading = skiprows_partitioning = skiprows_md = 0
    if isinstance(skiprows, int):
        skiprows_partitioning = skiprows
    elif is_list_like(skiprows) and len(skiprows) > 0:
        skiprows_md = np.sort(skiprows)
        if np.all(np.diff(skiprows_md) == 1):
            pre_reading = skiprows_md[0] - header_size if skiprows_md[0] > header_size else 0
            skiprows_partitioning = len(skiprows_md)
            skiprows_md = 0
        elif skiprows_md[0] > header_size:
            skiprows_md = skiprows_md - header_size
    elif callable(skiprows):

        def skiprows_func(x):
            return skiprows(x + header_size)
        skiprows_md = skiprows_func
    return (skiprows_md, pre_reading, skiprows_partitioning)