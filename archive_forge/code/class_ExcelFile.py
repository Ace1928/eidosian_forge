from __future__ import annotations
import csv
import inspect
import pathlib
import pickle
import warnings
from typing import (
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import (
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.readers import _c_parser_defaults
from modin.config import ExperimentalNumPyAPI
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, enable_logging
from modin.utils import (
@_inherit_docstrings(pandas.ExcelFile)
class ExcelFile(ClassLogger, pandas.ExcelFile):
    """
    Class for parsing tabular excel sheets into DataFrame objects.
    """
    _behave_like_pandas = False

    def _set_pandas_mode(self):
        self._behave_like_pandas = True

    def __getattribute__(self, item):
        if item in ['_set_pandas_mode', '_behave_like_pandas']:
            return object.__getattribute__(self, item)
        default_behaviors = ['__init__', '__class__']
        method = super(ExcelFile, self).__getattribute__(item)
        if not self._behave_like_pandas and item not in default_behaviors:
            if callable(method):

                def return_handler(*args, **kwargs):
                    """
                    Replace the default behavior of methods with inplace kwarg.

                    Returns
                    -------
                    A Modin DataFrame in place of a pandas DataFrame, or the same
                    return type as pandas.ExcelFile.

                    Notes
                    -----
                    This function will replace all of the arguments passed to
                    methods of ExcelFile with the pandas equivalent. It will convert
                    Modin DataFrame to pandas DataFrame, etc.
                    """
                    if item[0] != '_':
                        ErrorMessage.default_to_pandas('`{}`'.format(item))
                    args = [to_pandas(arg) if isinstance(arg, ModinObjects.DataFrame) else arg for arg in args]
                    kwargs = {k: to_pandas(v) if isinstance(v, ModinObjects.DataFrame) else v for k, v in kwargs.items()}
                    obj = super(ExcelFile, self).__getattribute__(item)(*args, **kwargs)
                    if isinstance(obj, pandas.DataFrame):
                        return ModinObjects.DataFrame(obj)
                    return obj
                method = return_handler
        return method