from __future__ import annotations
import numbers
import typing
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..exceptions import PlotnineError
class stage:
    """
    Stage allows you evaluating mapping at more than one stage

    You can evaluate an expression of a variable in a dataframe, and
    later evaluate an expression that modifies the values mapped to
    the scale.

    Parameters
    ----------
    start : str | array_like | scalar
        Aesthetic expression using primary variables from the layer
        data.
    after_stat : str
        Aesthetic expression using variables calculated by the stat.
    after_scale : str
        Aesthetic expression using aesthetics of the layer.
    """

    def __init__(self, start=None, after_stat=None, after_scale=None):
        self.start = start
        self.after_stat = after_stat
        self.after_scale = after_scale

    def __repr__(self):
        """
        Repr for staged mapping
        """
        if self.after_stat is None and self.after_scale is None:
            return f'{repr(self.start)}'
        if self.start is None and self.after_scale is None:
            return f'after_stat({repr(self.after_stat)})'
        if self.start is None and self.after_stat is None:
            return f'after_scale({repr(self.after_scale)})'
        return f'stage(start={repr(self.start)}, after_stat={repr(self.after_stat)}, after_scale={repr(self.after_scale)})'