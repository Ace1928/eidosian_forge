from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_number:
    """
    Labelling numbers

    Parameters
    ----------
    precision : int
        Number of digits after the decimal point.
    suffix : str
        What to put after the value.
    big_mark : str
        The thousands separator. This is usually
        a comma or a dot.
    decimal_mark : str
        What to use to separate the decimals digits.

    Examples
    --------
    >>> label_number()([.654, .8963, .1])
    ['0.65', '0.90', '0.10']
    >>> label_number(accuracy=0.0001)([.654, .8963, .1])
    ['0.6540', '0.8963', '0.1000']
    >>> label_number(precision=4)([.654, .8963, .1])
    ['0.6540', '0.8963', '0.1000']
    >>> label_number(prefix="$")([5, 24, -42])
    ['$5', '$24', '-$42']
    >>> label_number(suffix="s")([5, 24, -42])
    ['5s', '24s', '-42s']
    >>> label_number(big_mark="_")([1e3, 1e4, 1e5, 1e6])
    ['1_000', '10_000', '100_000', '1_000_000']
    >>> label_number(width=3)([1, 10, 100, 1000])
    ['  1', ' 10', '100', '1000']
    >>> label_number(align="^", width=5)([1, 10, 100, 1000])
    ['  1  ', ' 10  ', ' 100 ', '1000 ']
    >>> label_number(style_positive=" ")([5, 24, -42])
    [' 5', ' 24', '-42']
    >>> label_number(style_positive="+")([5, 24, -42])
    ['+5', '+24', '-42']
    >>> label_number(prefix="$", style_negative="braces")([5, 24, -42])
    ['$5', '$24', '($42)']
    """
    accuracy: Optional[float] = None
    precision: Optional[int] = None
    scale: float = 1
    prefix: str = ''
    suffix: str = ''
    big_mark: str = ''
    decimal_mark: str = '.'
    fill: str = ''
    style_negative: Literal['-', 'hyphen', 'parens'] = '-'
    style_positive: Literal['', '+', ' '] = ''
    align: Literal['<', '>', '=', '^'] = '>'
    width: Optional[int] = None

    def __post_init__(self):
        if self.precision is not None:
            if self.accuracy is not None:
                raise ValueError('Specify only one of precision or accuracy')
            self.accuracy = 10 ** (-self.precision)

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        valid_big_mark = self.big_mark in ('', ',', '_')
        sep = self.big_mark if valid_big_mark else ','
        fmt = f'{self.prefix}{{num:{sep}.{{precision}}f}}{self.suffix}'.format
        x = np.asarray(x)
        x_scaled = x * self.scale
        if self.accuracy is None:
            accuracy = precision(x_scaled)
        else:
            accuracy = self.accuracy
        x = round_any(x, accuracy / self.scale)
        digits = -np.floor(np.log10(accuracy)).astype(int)
        digits = np.minimum(np.maximum(digits, 0), 20)
        res = [fmt(num=abs(n), precision=digits) for n in x_scaled]
        if not valid_big_mark:
            res = [s.replace(',', self.big_mark) for s in res]
        if self.decimal_mark != '.':
            res = [s.replace('.', self.decimal_mark) for s in res]
        pos_fmt = f'{self.style_positive}{{s}}'.format
        if self.style_negative == '-':
            neg_fmt = '-{s}'.format
        elif self.style_negative == 'hyphen':
            neg_fmt = '−{s}'.format
        else:
            neg_fmt = '({s})'.format
        res = [neg_fmt(s=s) if num < 0 else pos_fmt(s=s) for num, s in zip(x, res)]
        if self.width is not None:
            fmt = f'{{s:{self.fill}{self.align}{self.width}}}'.format
            res = [fmt(s=s) for s in res]
        return res