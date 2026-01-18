from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddFlagPrefix(name):
    """Add the flag prefix to a name, if not present."""
    if name.startswith('--'):
        return name
    return '--' + name