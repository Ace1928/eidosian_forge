from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING
from ..util.sampledata import package_csv
def _capitalize_words(string: str) -> str:
    """

    """
    return ' '.join((word.capitalize() for word in string.split(' ')))