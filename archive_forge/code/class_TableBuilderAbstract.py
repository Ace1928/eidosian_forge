from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class TableBuilderAbstract(ABC):
    """
    Abstract table builder producing string representation of LaTeX table.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
        Instance of `DataFrameFormatter`.
    column_format: str, optional
        Column format, for example, 'rcl' for three columns.
    multicolumn: bool, optional
        Use multicolumn to enhance MultiIndex columns.
    multicolumn_format: str, optional
        The alignment for multicolumns, similar to column_format.
    multirow: bool, optional
        Use multirow to enhance MultiIndex rows.
    caption: str, optional
        Table caption.
    short_caption: str, optional
        Table short caption.
    label: str, optional
        LaTeX label.
    position: str, optional
        Float placement specifier, for example, 'htb'.
    """

    def __init__(self, formatter: DataFrameFormatter, column_format: str | None=None, multicolumn: bool=False, multicolumn_format: str | None=None, multirow: bool=False, caption: str | None=None, short_caption: str | None=None, label: str | None=None, position: str | None=None) -> None:
        self.fmt = formatter
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.caption = caption
        self.short_caption = short_caption
        self.label = label
        self.position = position

    def get_result(self) -> str:
        """String representation of LaTeX table."""
        elements = [self.env_begin, self.top_separator, self.header, self.middle_separator, self.env_body, self.bottom_separator, self.env_end]
        result = '\n'.join([item for item in elements if item])
        trailing_newline = '\n'
        result += trailing_newline
        return result

    @property
    @abstractmethod
    def env_begin(self) -> str:
        """Beginning of the environment."""

    @property
    @abstractmethod
    def top_separator(self) -> str:
        """Top level separator."""

    @property
    @abstractmethod
    def header(self) -> str:
        """Header lines."""

    @property
    @abstractmethod
    def middle_separator(self) -> str:
        """Middle level separator."""

    @property
    @abstractmethod
    def env_body(self) -> str:
        """Environment body."""

    @property
    @abstractmethod
    def bottom_separator(self) -> str:
        """Bottom level separator."""

    @property
    @abstractmethod
    def env_end(self) -> str:
        """End of the environment."""