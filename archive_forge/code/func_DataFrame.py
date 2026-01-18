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
@classproperty
def DataFrame(cls):
    """Get ``modin.pandas.DataFrame`` class."""
    if cls._dataframe is None:
        from .dataframe import DataFrame
        cls._dataframe = DataFrame
    return cls._dataframe