import sys
import os
import re
import warnings
import types
import unicodedata
class reprunicode(str):
    """
        A unicode sub-class that removes the initial u from unicode's repr.
        """

    def __repr__(self):
        return str.__repr__(self)[1:]