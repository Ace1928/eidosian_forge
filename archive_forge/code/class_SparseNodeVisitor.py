import sys
import os
import re
import warnings
import types
import unicodedata
class SparseNodeVisitor(NodeVisitor):
    """
    Base class for sparse traversals, where only certain node types are of
    interest.  When ``visit_...`` & ``depart_...`` methods should be
    implemented for *all* node types (such as for `docutils.writers.Writer`
    subclasses), subclass `NodeVisitor` instead.
    """