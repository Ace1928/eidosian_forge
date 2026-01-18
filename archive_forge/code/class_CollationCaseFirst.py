from __future__ import annotations
from typing import Any, Mapping, Optional, Union
from pymongo import common
class CollationCaseFirst:
    """
    An enum that defines values for `case_first` on a
    :class:`~pymongo.collation.Collation`.
    """
    UPPER = 'upper'
    'Sort uppercase characters first.'
    LOWER = 'lower'
    'Sort lowercase characters first.'
    OFF = 'off'
    'Default for locale or collation strength.'