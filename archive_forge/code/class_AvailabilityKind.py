from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class AvailabilityKind(BaseEnumeration):
    """
    Describes the availability of an entity.
    """
    _kinds = []
    _name_map = None

    def __repr__(self):
        return 'AvailabilityKind.%s' % (self.name,)