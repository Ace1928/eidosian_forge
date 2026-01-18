import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class Deb822Dict(_Deb822Dict_base):
    """A dictionary-like object suitable for storing RFC822-like data.

    Deb822Dict behaves like a normal dict, except:
        - key lookup is case-insensitive
        - key order is preserved
        - if initialized with a _parsed parameter, it will pull values from
          that dictionary-like object as needed (rather than making a copy).
          The _parsed dict is expected to be able to handle case-insensitive
          keys.

    If _parsed is not None, an optional _fields parameter specifies which keys
    in the _parsed dictionary are exposed.
    """

    def __init__(self, _dict=None, _parsed=None, _fields=None, encoding='utf-8'):
        self.__dict = {}
        self.__keys = OrderedSet()
        self.__parsed = None
        self.encoding = encoding
        self.decoder = _AutoDecoder(self.encoding)
        super(Deb822Dict, self).__init__()
        if _dict is not None:
            items = []
            if hasattr(_dict, 'items'):
                items = _dict.items()
            else:
                items = list(_dict)
            try:
                for k, v in items:
                    self[k] = v
            except ValueError:
                this = len(self.__keys)
                len_ = len(items[this])
                raise ValueError('dictionary update sequence element #%d has length %d; 2 is required' % (this, len_))
        if _parsed is not None:
            self.__parsed = _parsed
            if _fields is None:
                self.__keys.extend([_cached_strI(k) for k in self.__parsed])
            else:
                self.__keys.extend([_cached_strI(f) for f in _fields if f in self.__parsed])

    def __iter__(self):
        for key in self.__keys:
            yield str(key)

    def __len__(self):
        return len(self.__keys)

    def __setitem__(self, key, value):
        keyi = _cached_strI(key)
        self.__keys.add(keyi)
        self.__dict[keyi] = value

    def __getitem__(self, key):
        keyi = _strI(key)
        try:
            value = self.__dict[keyi]
        except KeyError:
            if self.__parsed is not None and keyi in self.__keys:
                value = self.__parsed[keyi]
            else:
                raise
        return self.decoder.decode(value)

    def __delitem__(self, key):
        keyi = _strI(key)
        self.__keys.remove(keyi)
        try:
            del self.__dict[keyi]
        except KeyError:
            pass

    def __contains__(self, key):
        keyi = _strI(key)
        return keyi in self.__keys

    def order_last(self, field):
        """Re-order the given field so it is "last" in the paragraph"""
        self.__keys.order_last(_strI(field))

    def order_first(self, field):
        """Re-order the given field so it is "first" in the paragraph"""
        self.__keys.order_first(_strI(field))

    def order_before(self, field, reference_field):
        """Re-order the given field so appears directly after the reference field in the paragraph

        The reference field must be present."""
        self.__keys.order_before(_strI(field), _strI(reference_field))

    def order_after(self, field, reference_field):
        """Re-order the given field so appears directly before the reference field in the paragraph

        The reference field must be present.
        """
        self.__keys.order_after(_strI(field), _strI(reference_field))

    def sort_fields(self, key=None):
        """Re-order all fields

        :param key: Provide a key function (same semantics as for sorted).  Keep in mind that
          Deb822 preserve the cases for field names - in generally, callers are recommended to use
          "lower()" to normalize the case.
        """
        if key is None:
            key = default_field_sort_key
        self.__keys = OrderedSet(sorted(self.__keys, key=key))

    def __repr__(self):
        return '{%s}' % ', '.join(['%r: %r' % (k, v) for k, v in self.items()])

    def __eq__(self, other):
        mykeys = sorted(self)
        otherkeys = sorted(other)
        if not mykeys == otherkeys:
            return False
        for key in mykeys:
            if self[key] != other[key]:
                return False
        return True
    __hash__ = None

    def copy(self):
        copy = self.__class__(self)
        return copy