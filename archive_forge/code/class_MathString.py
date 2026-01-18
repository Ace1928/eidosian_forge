from __future__ import annotations
import logging # isort:skip
import re
from typing import Any
from .bases import Init
from .primitive import String
from .singletons import Undefined
class MathString(String):
    """ A string with math TeX/LaTeX delimiters.

    Args:
        value : a string that contains math

    """