import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXVariantGroup(PBXGroup, XCFileLikeElement):
    """PBXVariantGroup is used by Xcode to represent localizations."""
    pass