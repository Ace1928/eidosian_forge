import pandas
from pandas.io.common import stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher

        Read data from `filepath_or_buffer` according to the passed `read_custom_text` `kwargs` parameters.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_custom_text` function.
        columns : list or callable(file-like object, \*\*kwargs -> list
            Column names of list type or callable that create column names from opened file
            and passed `kwargs`.
        custom_parser : callable(file-like object, \*\*kwargs -> pandas.DataFrame
            Function that takes as input a part of the `filepath_or_buffer` file loaded into
            memory in file-like object form.
        **kwargs : dict
            Parameters of `read_custom_text` function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        