import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _AddChildToDicts(self, child):
    child_path = child.PathFromSourceTreeAndPath()
    if child_path:
        if child_path in self._children_by_path:
            raise ValueError('Found multiple children with path ' + child_path)
        self._children_by_path[child_path] = child
    if isinstance(child, PBXVariantGroup):
        child_name = child._properties.get('name', None)
        key = (child_name, child_path)
        if key in self._variant_children_by_name_and_path:
            raise ValueError('Found multiple PBXVariantGroup children with ' + 'name ' + str(child_name) + ' and path ' + str(child_path))
        self._variant_children_by_name_and_path[key] = child