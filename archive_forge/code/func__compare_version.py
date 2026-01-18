import re
def _compare_version(self, other):
    """Compare major.minor.bugfix"""
    if self.major == other.major:
        if self.minor == other.minor:
            if self.bugfix == other.bugfix:
                vercmp = 0
            elif self.bugfix > other.bugfix:
                vercmp = 1
            else:
                vercmp = -1
        elif self.minor > other.minor:
            vercmp = 1
        else:
            vercmp = -1
    elif self.major > other.major:
        vercmp = 1
    else:
        vercmp = -1
    return vercmp