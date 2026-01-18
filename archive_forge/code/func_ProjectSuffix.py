from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ProjectSuffix(uri):
    """Get the entire relative path of the object the uri refers to."""
    return uri.split('/projects/')[-1]