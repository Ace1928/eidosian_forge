from __future__ import annotations
from typing import Any, Mapping, Optional, Union
from pymongo import common
class CollationStrength:
    """
    An enum that defines values for `strength` on a
    :class:`~pymongo.collation.Collation`.
    """
    PRIMARY = 1
    'Differentiate base (unadorned) characters.'
    SECONDARY = 2
    'Differentiate character accents.'
    TERTIARY = 3
    'Differentiate character case.'
    QUATERNARY = 4
    'Differentiate words with and without punctuation.'
    IDENTICAL = 5
    'Differentiate unicode code point (characters are exactly identical).'