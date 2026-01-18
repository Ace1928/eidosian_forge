from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class TabularBuilder(GenericTableBuilder):
    """Concrete table builder for tabular environment.

    >>> from pandas.io.formats import format as fmt
    >>> df = pd.DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = TabularBuilder(formatter, column_format='lrc')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{tabular}{lrc}
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\bottomrule
    \\end{tabular}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str:
        return f'\\begin{{tabular}}{{{self.column_format}}}'

    @property
    def bottom_separator(self) -> str:
        return '\\bottomrule'

    @property
    def env_end(self) -> str:
        return '\\end{tabular}'