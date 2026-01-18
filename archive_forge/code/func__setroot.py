import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _setroot(self, element):
    """Replace root element of this tree.

        This will discard the current contents of the tree and replace it
        with the given element.  Use with care!

        """
    self._root = element