from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def StripFlagPrefix(name):
    """Strip the flag prefix from a name, if present."""
    if name.startswith('--'):
        return name[2:]
    return name