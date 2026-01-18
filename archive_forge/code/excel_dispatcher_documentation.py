import os
import re
import warnings
from io import BytesIO
import pandas
from pandas.io.common import stringify_path
from modin.config import NPartitions
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from modin.pandas.io import ExcelFile

        Read data from `io` according to the passed `read_excel` `kwargs` parameters.

        Parameters
        ----------
        io : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
            `io` parameter of `read_excel` function.
        **kwargs : dict
            Parameters of `read_excel` function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        