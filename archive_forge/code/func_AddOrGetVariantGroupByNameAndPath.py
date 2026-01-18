import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def AddOrGetVariantGroupByNameAndPath(self, name, path):
    """Returns an existing or new PBXVariantGroup for name and path.

    If a PBXVariantGroup identified by the name and path arguments is already
    present as a child of this object, it is returned.  Otherwise, a new
    PBXVariantGroup with the correct properties is created, added as a child,
    and returned.

    This method will generally be called by AddOrGetFileByPath, which knows
    when to create a variant group based on the structure of the pathnames
    passed to it.
    """
    key = (name, path)
    if key in self._variant_children_by_name_and_path:
        variant_group_ref = self._variant_children_by_name_and_path[key]
        assert variant_group_ref.__class__ == PBXVariantGroup
        return variant_group_ref
    variant_group_properties = {'name': name}
    if path is not None:
        variant_group_properties['path'] = path
    variant_group_ref = PBXVariantGroup(variant_group_properties)
    self.AppendChild(variant_group_ref)
    return variant_group_ref