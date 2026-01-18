import sys
import os
import re
import warnings
import types
import unicodedata
class NodeFound(TreePruningException):
    """
    Raise to indicate that the target of a search has been found.  This
    exception must be caught by the client; it is not caught by the traversal
    code.
    """
    pass