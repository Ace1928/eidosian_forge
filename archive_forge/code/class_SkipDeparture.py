import sys
import os
import re
import warnings
import types
import unicodedata
class SkipDeparture(TreePruningException):
    """
    Do not call the current node's ``depart_...`` method.  The current node's
    children and siblings are not affected.
    """
    pass