from __future__ import annotations
import typing
from .exceptions import PlotnineError
from .iapi import labels_view
from .mapping.aes import SCALED_AESTHETICS, rename_aesthetics
class ggtitle(labs):
    """
    Create plot title

    Parameters
    ----------
    title :
        Plot title
    """

    def __init__(self, title: str):
        if title is None:
            raise PlotnineError('Arguments to ggtitle cannot be None')
        self.labels = labels_view(title=title)