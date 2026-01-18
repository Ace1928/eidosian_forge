import re
from . import errors, osutils, transport
class ViewsNotSupported(errors.BzrError):
    """Views are not supported by a tree format.
    """
    _fmt = "Views are not supported by %(tree)s; use 'brz upgrade' to change your tree to a later format."

    def __init__(self, tree):
        self.tree = tree