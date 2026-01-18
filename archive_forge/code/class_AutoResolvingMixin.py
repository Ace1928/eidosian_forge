import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
class AutoResolvingMixin(Generic[T], _ParagraphMapping_Base[T]):

    @property
    def _auto_resolve_ambiguous_fields(self):
        return True

    @property
    def _paragraph(self):
        raise NotImplementedError

    def __len__(self):
        return self._paragraph.kvpair_count

    def __contains__(self, item):
        return self._paragraph.contains_kvpair_element(item)

    def __iter__(self):
        return iter(self._paragraph.iter_keys())

    def __getitem__(self, item):
        if self._auto_resolve_ambiguous_fields and isinstance(item, str):
            v = self._paragraph.get_kvpair_element((item, 0))
        else:
            v = self._paragraph.get_kvpair_element(item)
        assert v is not None
        return self._interpret_value(item, v)

    def __delitem__(self, item):
        self._paragraph.remove_kvpair_element(item)

    def _interpret_value(self, key, value):
        raise NotImplementedError