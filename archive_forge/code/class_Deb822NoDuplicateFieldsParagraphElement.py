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
class Deb822NoDuplicateFieldsParagraphElement(Deb822ParagraphElement):
    """Paragraph implementation optimized for valid deb822 files

    When there are no duplicated fields, we can use simpler and faster
    datastructures for common operations.
    """

    def __init__(self, kvpair_elements, kvpair_order):
        super().__init__()
        self._kvpair_elements = {kv.field_name: kv for kv in kvpair_elements}
        self._kvpair_order = kvpair_order
        self._init_parent_of_parts()

    @property
    def kvpair_count(self):
        return len(self._kvpair_elements)

    def order_last(self, field):
        """Re-order the given field so it is "last" in the paragraph"""
        unpacked_field, _, _ = _unpack_key(field, raise_if_indexed=True)
        self._kvpair_order.order_last(unpacked_field)

    def order_first(self, field):
        """Re-order the given field so it is "first" in the paragraph"""
        unpacked_field, _, _ = _unpack_key(field, raise_if_indexed=True)
        self._kvpair_order.order_first(unpacked_field)

    def order_before(self, field, reference_field):
        """Re-order the given field so appears directly after the reference field in the paragraph

        The reference field must be present."""
        unpacked_field, _, _ = _unpack_key(field, raise_if_indexed=True)
        unpacked_ref_field, _, _ = _unpack_key(reference_field, raise_if_indexed=True)
        self._kvpair_order.order_before(unpacked_field, unpacked_ref_field)

    def order_after(self, field, reference_field):
        """Re-order the given field so appears directly before the reference field in the paragraph

        The reference field must be present.
        """
        unpacked_field, _, _ = _unpack_key(field, raise_if_indexed=True)
        unpacked_ref_field, _, _ = _unpack_key(reference_field, raise_if_indexed=True)
        self._kvpair_order.order_after(unpacked_field, unpacked_ref_field)

    def __iter__(self):
        return iter((str(k) for k in self._kvpair_order))

    def iter_keys(self):
        yield from (str(k) for k in self._kvpair_order)

    def remove_kvpair_element(self, key):
        key, _, _ = _unpack_key(key, raise_if_indexed=True)
        del self._kvpair_elements[key]
        self._kvpair_order.remove(key)

    def contains_kvpair_element(self, item):
        if not isinstance(item, (str, tuple, Deb822FieldNameToken)):
            return False
        item = cast('ParagraphKey', item)
        key, _, _ = _unpack_key(item, raise_if_indexed=True)
        return key in self._kvpair_elements

    def get_kvpair_element(self, item, use_get=False):
        item, _, _ = _unpack_key(item, raise_if_indexed=True)
        if use_get:
            return self._kvpair_elements.get(item)
        return self._kvpair_elements[item]

    def set_kvpair_element(self, key, value):
        key, _, _ = _unpack_key(key, raise_if_indexed=True)
        if isinstance(key, Deb822FieldNameToken):
            if key is not value.field_token:
                raise ValueError('Key is a Deb822FieldNameToken, but not *the* Deb822FieldNameToken for the value')
            key = value.field_name
        else:
            if key != value.field_name:
                raise ValueError('Cannot insert value under a different field value than field name from its Deb822FieldNameToken implies')
            key = value.field_name
        original_value = self._kvpair_elements.get(key)
        self._kvpair_elements[key] = value
        self._kvpair_order.append(key)
        if original_value is not None:
            original_value.parent_element = None
        value.parent_element = self

    def sort_fields(self, key=None):
        """Re-order all fields

        :param key: Provide a key function (same semantics as for sorted).  Keep in mind that
          the module preserve the cases for field names - in generally, callers are recommended
          to use "lower()" to normalize the case.
        """
        for last_field_name in reversed(self._kvpair_order):
            last_kvpair = self._kvpair_elements[cast('_strI', last_field_name)]
            last_kvpair.value_element.add_final_newline_if_missing()
            break
        if key is None:
            key = default_field_sort_key
        self._kvpair_order = OrderedSet(sorted(self._kvpair_order, key=key))

    def iter_parts(self):
        yield from (self._kvpair_elements[x] for x in cast('Iterable[_strI]', self._kvpair_order))