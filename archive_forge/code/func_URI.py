from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def URI(ref):
    """Converts a resource reference into its URI representation."""
    return ref.SelfLink() if ref else None