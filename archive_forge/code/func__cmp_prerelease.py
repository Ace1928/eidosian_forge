import contextlib
import re
import warnings
def _cmp_prerelease(self, other):
    """
        case 1: self has prerelease, other doesn't; other is greater
        case 2: self doesn't have prerelease, other does: self is greater
        case 3: both or neither have prerelease: compare them!
        """
    if self.prerelease and (not other.prerelease):
        return -1
    elif not self.prerelease and other.prerelease:
        return 1
    if self.prerelease == other.prerelease:
        return 0
    elif self.prerelease < other.prerelease:
        return -1
    else:
        return 1