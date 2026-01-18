from __future__ import annotations
import functools
import warnings
from textwrap import dedent
from typing import Optional, Type, Union
class PlotnineWarning(UserWarning):
    """
    Warnings for ggplot inconsistencies
    """