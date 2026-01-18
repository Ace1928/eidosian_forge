import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class AppliedPatches:
    """Context that provides access to a tree with patches applied.
    """

    def __init__(self, tree, patches, prefix=1):
        self.tree = tree
        self.patches = patches
        self.prefix = prefix

    def __enter__(self):
        self._tt = self.tree.preview_transform()
        apply_patches(self._tt, self.patches, prefix=self.prefix)
        return self._tt.get_preview_tree()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._tt.finalize()
        return False