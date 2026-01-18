from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
def _tidyup_labels(self, labels: Sequence[str]) -> Sequence[str]:
    """
        Make all labels uniform in format

        Remove redundant zeros for labels in exponential format.

        Parameters
        ----------
        labels : list-like
            Labels to be tidied.

        Returns
        -------
        out : list-like
            Labels
        """

    def remove_zeroes(s: str) -> str:
        """
            Remove unnecessary zeros for float string s
            """
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            s = f'{mantissa}e{exponent}' if exponent else mantissa
        return s

    def as_exp(s: str) -> str:
        """
            Float string s as in exponential format
            """
        return s if 'e' in s else '{:1.0e}'.format(float(s))

    def as_mathtex(s: str) -> str:
        """
            Mathtex for maplotlib
            """
        if 'e' not in s:
            assert s == '1', f"Unexpected value s = {s!r}, instead of '1'"
            return f'${self.base}^{{0}}$'
        exp = s.split('e')[1]
        return f'${self.base}^{{{exp}}}$'
    has_e = ['e' in x for x in labels]
    if not all(has_e) and sum(has_e):
        labels = [as_exp(x) for x in labels]
    labels = [remove_zeroes(x) for x in labels]
    has_e = ['e' in x for x in labels]
    if self.mathtex and any(has_e):
        labels = [as_mathtex(x) for x in labels]
    return labels