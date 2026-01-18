import sys
import os
import re
import warnings
import types
import unicodedata
class SkipSiblings(TreePruningException):
    """
    Do not visit any more siblings (to the right) of the current node.  The
    current node's children and its ``depart_...`` method are not affected.
    """
    pass