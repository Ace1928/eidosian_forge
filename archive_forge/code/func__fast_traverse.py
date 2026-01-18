import sys
import os
import re
import warnings
import types
import unicodedata
def _fast_traverse(self, cls):
    """Specialized traverse() that only supports instance checks."""
    result = []
    if isinstance(self, cls):
        result.append(self)
    for child in self.children:
        result.extend(child._fast_traverse(cls))
    return result