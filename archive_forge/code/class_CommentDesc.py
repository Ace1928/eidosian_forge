from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
class CommentDesc:
    """Describe how to extract a metadata entry from a comment string"""
    n: int
    'Which of the 5 SPE comment fields to use.'
    slice: slice
    'Which characters from the `n`-th comment to use.'
    cvt: Callable[[str], Any]
    'How to convert characters to something useful.'
    scale: Union[None, float]
    'Optional scaling factor for numbers'

    def __init__(self, n: int, slice: slice, cvt: Callable[[str], Any]=str, scale: Optional[float]=None):
        self.n = n
        self.slice = slice
        self.cvt = cvt
        self.scale = scale