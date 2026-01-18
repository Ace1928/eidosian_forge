from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class RefQualifierKind(BaseEnumeration):
    """Describes a specific ref-qualifier of a type."""
    _kinds = []
    _name_map = None

    def from_param(self):
        return self.value

    def __repr__(self):
        return 'RefQualifierKind.%s' % (self.name,)