import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def Descendants(self):
    """Returns a list of all of this object's descendants, including this
    object.
    """
    children = self.Children()
    descendants = [self]
    for child in children:
        descendants.extend(child.Descendants())
    return descendants