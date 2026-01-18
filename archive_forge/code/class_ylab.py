from __future__ import annotations
import typing
from .exceptions import PlotnineError
from .iapi import labels_view
from .mapping.aes import SCALED_AESTHETICS, rename_aesthetics
class ylab(labs):
    """
    Create y-axis label

    Parameters
    ----------
    ylab :
        y-axis label
    """

    def __init__(self, ylab: str):
        if ylab is None:
            raise PlotnineError('Arguments to ylab cannot be None')
        self.labels = labels_view(y=ylab)