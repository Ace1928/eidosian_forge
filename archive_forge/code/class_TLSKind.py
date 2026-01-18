from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class TLSKind(BaseEnumeration):
    """Describes the kind of thread-local storage (TLS) of a cursor."""
    _kinds = []
    _name_map = None

    def from_param(self):
        return self.value

    def __repr__(self):
        return 'TLSKind.%s' % (self.name,)