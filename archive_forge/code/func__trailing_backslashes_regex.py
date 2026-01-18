import re
from . import lazy_regex
from .trace import mutter, warning
def _trailing_backslashes_regex(m):
    """Check trailing backslashes.

    Does a head count on trailing backslashes to ensure there isn't an odd
    one on the end that would escape the brackets we wrap the RE in.
    """
    if len(m) % 2 != 0:
        warning("Regular expressions cannot end with an odd number of '\\'. Dropping the final '\\'.")
        return m[:-1]
    return m