from typing import List, Optional, Union
from . import errors, hooks, osutils, trace, tree
class BadReferenceTarget(errors.InternalBzrError):
    _fmt = "Can't add reference to %(other_tree)s into %(tree)s.%(reason)s"

    def __init__(self, tree, other_tree, reason):
        self.tree = tree
        self.other_tree = other_tree
        self.reason = reason