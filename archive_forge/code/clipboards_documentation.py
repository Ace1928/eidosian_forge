from __future__ import annotations
from io import StringIO
from typing import TYPE_CHECKING
import warnings
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.generic import ABCDataFrame
from pandas import (

    Attempt to write text representation of object to the system clipboard
    The clipboard can be then pasted into Excel for example.

    Parameters
    ----------
    obj : the object to write to the clipboard
    excel : bool, defaults to True
            if True, use the provided separator, writing in a csv
            format for allowing easy pasting into excel.
            if False, write a string representation of the object
            to the clipboard
    sep : optional, defaults to tab
    other keywords are passed to to_csv

    Notes
    -----
    Requirements for your platform
      - Linux: xclip, or xsel (with PyQt4 modules)
      - Windows:
      - OS X:
    