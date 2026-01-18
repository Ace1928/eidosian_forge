import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def VerifyHasRequiredProperties(self):
    """Ensure that all properties identified as required by the schema are
    set.
    """
    for property, attributes in self._schema.items():
        is_list, property_type, is_strong, is_required = attributes[0:4]
        if is_required and property not in self._properties:
            raise KeyError(self.__class__.__name__ + ' requires ' + property)