import sys
import os
import re
import warnings
import types
import unicodedata
class SkipNode(TreePruningException):
    """
    Do not visit the current node's children, and do not call the current
    node's ``depart_...`` method.
    """
    pass