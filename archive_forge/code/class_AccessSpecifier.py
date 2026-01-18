from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class AccessSpecifier(BaseEnumeration):
    """
    Describes the access of a C++ class member
    """
    _kinds = []
    _name_map = None

    def from_param(self):
        return self.value

    def __repr__(self):
        return 'AccessSpecifier.%s' % (self.name,)