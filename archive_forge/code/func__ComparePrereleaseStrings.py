from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from six.moves import zip_longest
@classmethod
def _ComparePrereleaseStrings(cls, s1, s2):
    """Compares the two given prerelease strings.

    Args:
      s1: str, The first prerelease string.
      s2: str, The second prerelease string.

    Returns:
      1 if s1 is greater than s2, -1 if s2 is greater than s1, and 0 if equal.
    """
    s1 = s1.split('.') if s1 else []
    s2 = s2.split('.') if s2 else []
    for this, other in zip_longest(s1, s2):
        if this is None:
            return 1
        elif other is None:
            return -1
        if this == other:
            continue
        if this.isdigit() and other.isdigit():
            return SemVer._CmpHelper(int(this), int(other))
        return SemVer._CmpHelper(this.lower(), other.lower())
    return 0