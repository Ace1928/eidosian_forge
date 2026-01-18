from __future__ import division
import sys
import unicodedata
from functools import reduce
class FallbackToText(Exception):
    """Used for failed conversion to float"""
    pass