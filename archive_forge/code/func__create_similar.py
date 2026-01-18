from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def _create_similar(self, parts):
    """
        Create a new text object of the same type with the same parameters,
        with different text content.

        >>> text = Tag('strong', 'Bananas!')
        >>> text._create_similar(['Apples!'])
        Tag('strong', 'Apples!')
        """
    cls, cls_args = self._typeinfo()
    args = list(cls_args) + list(parts)
    return cls(*args)