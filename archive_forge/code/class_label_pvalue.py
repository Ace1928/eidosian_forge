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
class label_pvalue:
    """
    p-values labelling

    Parameters
    ----------
    accuracy : float
        Number to round to
    add_p : bool
        Whether to prepend "p=" or "p<" to the output

    Examples
    --------
    >>> x = [.90, .15, .015, .009, 0.0005]
    >>> label_pvalue()(x)
    ['0.9', '0.15', '0.015', '0.009', '<0.001']
    >>> label_pvalue(0.1)(x)
    ['0.9', '0.1', '<0.1', '<0.1', '<0.1']
    >>> label_pvalue(0.1, True)(x)
    ['p=0.9', 'p=0.1', 'p<0.1', 'p<0.1', 'p<0.1']
    """
    accuracy: float = 0.001
    add_p: float = False

    def __call__(self, x: FloatArrayLike) -> Sequence[str]:
        """
        Format a sequence of inputs

        Parameters
        ----------
        x : array
            Input

        Returns
        -------
        out : list
            List of strings.
        """
        x = round_any(x, self.accuracy)
        below = [num < self.accuracy for num in x]
        if self.add_p:
            eq_fmt = 'p={:g}'.format
            below_label = f'p<{self.accuracy:g}'
        else:
            eq_fmt = '{:g}'.format
            below_label = f'<{self.accuracy:g}'
        labels = [below_label if b else eq_fmt(i) for i, b in zip(x, below)]
        return labels