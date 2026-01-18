from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def _merge_similar(self, parts):
    """Merge adjacent text objects with the same type and parameters together.

        >>> text = Text()
        >>> parts = [Tag('em', 'Breaking'), Tag('em', ' '), Tag('em', 'news!')]
        >>> list(text._merge_similar(parts))
        [Tag('em', 'Breaking news!')]
        """
    groups = itertools.groupby(parts, lambda value: value._typeinfo())
    for typeinfo, group in groups:
        cls, info = typeinfo
        group = list(group)
        if cls and len(group) > 1:
            group_parts = itertools.chain(*(text.parts for text in group))
            args = list(info) + list(group_parts)
            yield cls(*args)
        else:
            for text in group:
                yield text