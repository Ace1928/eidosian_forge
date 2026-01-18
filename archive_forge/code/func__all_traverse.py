import sys
import os
import re
import warnings
import types
import unicodedata
def _all_traverse(self):
    """Specialized traverse() that doesn't check for a condition."""
    result = []
    result.append(self)
    for child in self.children:
        result.extend(child._all_traverse())
    return result