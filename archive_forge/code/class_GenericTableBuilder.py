from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class GenericTableBuilder(TableBuilderAbstract):
    """Table builder producing string representation of LaTeX table."""

    @property
    def header(self) -> str:
        iterator = self._create_row_iterator(over='header')
        return '\n'.join(list(iterator))

    @property
    def top_separator(self) -> str:
        return '\\toprule'

    @property
    def middle_separator(self) -> str:
        return '\\midrule' if self._is_separator_required() else ''

    @property
    def env_body(self) -> str:
        iterator = self._create_row_iterator(over='body')
        return '\n'.join(list(iterator))

    def _is_separator_required(self) -> bool:
        return bool(self.header and self.env_body)

    @property
    def _position_macro(self) -> str:
        """Position macro, extracted from self.position, like [h]."""
        return f'[{self.position}]' if self.position else ''

    @property
    def _caption_macro(self) -> str:
        """Caption macro, extracted from self.caption.

        With short caption:
            \\caption[short_caption]{caption_string}.

        Without short caption:
            \\caption{caption_string}.
        """
        if self.caption:
            return ''.join(['\\caption', f'[{self.short_caption}]' if self.short_caption else '', f'{{{self.caption}}}'])
        return ''

    @property
    def _label_macro(self) -> str:
        """Label macro, extracted from self.label, like \\label{ref}."""
        return f'\\label{{{self.label}}}' if self.label else ''

    def _create_row_iterator(self, over: str) -> RowStringIterator:
        """Create iterator over header or body of the table.

        Parameters
        ----------
        over : {'body', 'header'}
            Over what to iterate.

        Returns
        -------
        RowStringIterator
            Iterator over body or header.
        """
        iterator_kind = self._select_iterator(over)
        return iterator_kind(formatter=self.fmt, multicolumn=self.multicolumn, multicolumn_format=self.multicolumn_format, multirow=self.multirow)

    def _select_iterator(self, over: str) -> type[RowStringIterator]:
        """Select proper iterator over table rows."""
        if over == 'header':
            return RowHeaderIterator
        elif over == 'body':
            return RowBodyIterator
        else:
            msg = f"'over' must be either 'header' or 'body', but {over} was provided"
            raise ValueError(msg)