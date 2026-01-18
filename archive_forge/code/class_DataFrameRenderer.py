from __future__ import annotations
from collections.abc import (
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
import numpy as np
from pandas._config.config import (
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import (
from pandas.io.formats import printing
class DataFrameRenderer:
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.core.frame.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """

    def __init__(self, fmt: DataFrameFormatter) -> None:
        self.fmt = fmt

    def to_html(self, buf: FilePath | WriteBuffer[str] | None=None, encoding: str | None=None, classes: str | list | tuple | None=None, notebook: bool=False, border: int | bool | None=None, table_id: str | None=None, render_links: bool=False) -> str | None:
        """
        Render a DataFrame to a html table.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            ``<table>`` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        """
        from pandas.io.formats.html import HTMLFormatter, NotebookFormatter
        Klass = NotebookFormatter if notebook else HTMLFormatter
        html_formatter = Klass(self.fmt, classes=classes, border=border, table_id=table_id, render_links=render_links)
        string = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_string(self, buf: FilePath | WriteBuffer[str] | None=None, encoding: str | None=None, line_width: int | None=None) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding: str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width to wrap a line in characters.
        """
        from pandas.io.formats.string import StringFormatter
        string_formatter = StringFormatter(self.fmt, line_width=line_width)
        string = string_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_csv(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None=None, encoding: str | None=None, sep: str=',', columns: Sequence[Hashable] | None=None, index_label: IndexLabel | None=None, mode: str='w', compression: CompressionOptions='infer', quoting: int | None=None, quotechar: str='"', lineterminator: str | None=None, chunksize: int | None=None, date_format: str | None=None, doublequote: bool=True, escapechar: str | None=None, errors: str='strict', storage_options: StorageOptions | None=None) -> str | None:
        """
        Render dataframe as comma-separated file.
        """
        from pandas.io.formats.csvs import CSVFormatter
        if path_or_buf is None:
            created_buffer = True
            path_or_buf = StringIO()
        else:
            created_buffer = False
        csv_formatter = CSVFormatter(path_or_buf=path_or_buf, lineterminator=lineterminator, sep=sep, encoding=encoding, errors=errors, compression=compression, quoting=quoting, cols=columns, index_label=index_label, mode=mode, chunksize=chunksize, quotechar=quotechar, date_format=date_format, doublequote=doublequote, escapechar=escapechar, storage_options=storage_options, formatter=self.fmt)
        csv_formatter.save()
        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content = path_or_buf.getvalue()
            path_or_buf.close()
            return content
        return None