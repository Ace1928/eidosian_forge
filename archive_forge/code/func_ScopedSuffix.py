from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ScopedSuffix(uri):
    """Get just the scoped part of the object the uri refers to."""
    if '/zones/' in uri:
        return uri.split('/zones/')[-1]
    elif '/regions/' in uri:
        return uri.split('/regions/')[-1]
    else:
        return Name(uri)