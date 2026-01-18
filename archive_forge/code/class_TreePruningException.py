import sys
import os
import re
import warnings
import types
import unicodedata
class TreePruningException(Exception):
    """
    Base class for `NodeVisitor`-related tree pruning exceptions.

    Raise subclasses from within ``visit_...`` or ``depart_...`` methods
    called from `Node.walk()` and `Node.walkabout()` tree traversals to prune
    the tree traversed.
    """
    pass