from __future__ import (absolute_import, division, print_function)
import sys
import types
import warnings
from sys import intern as _sys_intern
from collections.abc import Mapping, Set
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.utils.native_jinja import NativeJinjaText
class AnsibleUnsafeBytes(bytes, AnsibleUnsafe):

    def _strip_unsafe(self):
        return super().__bytes__()

    def __reduce__(self, /):
        return (self.__class__, (self._strip_unsafe(),))

    def __str__(self, /):
        return self.decode()

    def __bytes__(self, /):
        return self

    def __repr__(self, /):
        return AnsibleUnsafeText(super().__repr__())

    def __format__(self, format_spec, /):
        return AnsibleUnsafeText(super().__format__(format_spec))

    def __getitem__(self, key, /):
        if isinstance(key, int):
            return super().__getitem__(key)
        return self.__class__(super().__getitem__(key))

    def __reversed__(self, /):
        return self[::-1]

    def __add__(self, value, /):
        return self.__class__(super().__add__(value))

    def __radd__(self, value, /):
        return self.__class__(value.__add__(self))

    def __mul__(self, value, /):
        return self.__class__(super().__mul__(value))
    __rmul__ = __mul__

    def __mod__(self, value, /):
        return self.__class__(super().__mod__(value))

    def __rmod__(self, value, /):
        return self.__class__(super().__rmod__(value))

    def capitalize(self, /):
        return self.__class__(super().capitalize())

    def center(self, width, fillchar=b' ', /):
        return self.__class__(super().center(width, fillchar))

    def decode(self, /, encoding='utf-8', errors='strict'):
        return AnsibleUnsafeText(super().decode(encoding=encoding, errors=errors))

    def removeprefix(self, prefix, /):
        return self.__class__(super().removeprefix(prefix))

    def removesuffix(self, suffix, /):
        return self.__class__(super().removesuffix(suffix))

    def expandtabs(self, /, tabsize=8):
        return self.__class__(super().expandtabs(tabsize))

    def join(self, iterable_of_bytes, /):
        return self.__class__(super().join(iterable_of_bytes))

    def ljust(self, width, fillchar=b' ', /):
        return self.__class__(super().ljust(width, fillchar))

    def lower(self, /):
        return self.__class__(super().lower())

    def lstrip(self, chars=None, /):
        return self.__class__(super().lstrip(chars))

    def partition(self, sep, /):
        cls = self.__class__
        return tuple((cls(e) for e in super().partition(sep)))

    def replace(self, old, new, count=-1, /):
        return self.__class__(super().replace(old, new, count))

    def rjust(self, width, fillchar=b' ', /):
        return self.__class__(super().rjust(width, fillchar))

    def rpartition(self, sep, /):
        cls = self.__class__
        return tuple((cls(e) for e in super().rpartition(sep)))

    def rstrip(self, chars=None, /):
        return self.__class__(super().rstrip(chars))

    def split(self, /, sep=None, maxsplit=-1):
        cls = self.__class__
        return [cls(e) for e in super().split(sep=sep, maxsplit=maxsplit)]

    def rsplit(self, /, sep=None, maxsplit=-1):
        cls = self.__class__
        return [cls(e) for e in super().rsplit(sep=sep, maxsplit=maxsplit)]

    def splitlines(self, /, keepends=False):
        cls = self.__class__
        return [cls(e) for e in super().splitlines(keepends=keepends)]

    def strip(self, chars=None, /):
        return self.__class__(super().strip(chars))

    def swapcase(self, /):
        return self.__class__(super().swapcase())

    def title(self, /):
        return self.__class__(super().title())

    def translate(self, table, /, delete=b''):
        return self.__class__(super().translate(table, delete=delete))

    def upper(self, /):
        return self.__class__(super().upper())

    def zfill(self, width, /):
        return self.__class__(super().zfill(width))