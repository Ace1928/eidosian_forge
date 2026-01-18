from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def Decode(s):
    """Return text objects decoded from HTTP headers / payload."""
    if s is None:
        return s
    return s.decode('utf-8')