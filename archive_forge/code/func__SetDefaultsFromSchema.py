import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _SetDefaultsFromSchema(self):
    """Assign object default values according to the schema.  This will not
    overwrite properties that have already been set."""
    defaults = {}
    for property, attributes in self._schema.items():
        is_list, property_type, is_strong, is_required = attributes[0:4]
        if is_required and len(attributes) >= 5 and (property not in self._properties):
            default = attributes[4]
            defaults[property] = default
    if len(defaults) > 0:
        self.UpdateProperties(defaults, do_copy=True)