import csv
import glob
import os
import warnings
from contextlib import ExitStack
from typing import List, Tuple
import fsspec
import pandas
import pandas._libs.lib as lib
from pandas.io.common import is_fsspec_url, is_url, stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.csv_dispatcher import CSVDispatcher

        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        files : file or list of files
            File(s) to be partitioned.
        fnames : str or list of str
            File name(s) to be partitioned.
        num_partitions : int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        nrows : int, optional
            Number of rows of file to read.
        skiprows : int, optional
            Specifies rows to skip.
        skip_header : int, optional
            Specifies header rows to skip.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.

        Returns
        -------
        list
            List, where each element of the list is a list of tuples. The inner lists
            of tuples contains the data file name of the chunk, chunk start offset, and
            chunk end offsets for its corresponding file.

        Notes
        -----
        The logic gets really complicated if we try to use the `TextFileDispatcher.partitioned_file`.
        