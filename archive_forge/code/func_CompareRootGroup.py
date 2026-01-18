import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def CompareRootGroup(self, other):
    order = ['Source', 'Intermediates', 'Projects', 'Frameworks', 'Products', 'Build']
    self_name = self.Name()
    other_name = other.Name()
    self_in = isinstance(self, PBXGroup) and self_name in order
    other_in = isinstance(self, PBXGroup) and other_name in order
    if not self_in and (not other_in):
        return self.Compare(other)
    if self_name in order and other_name not in order:
        return -1
    if other_name in order and self_name not in order:
        return 1
    self_index = order.index(self_name)
    other_index = order.index(other_name)
    if self_index < other_index:
        return -1
    if self_index > other_index:
        return 1
    return 0