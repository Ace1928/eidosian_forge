from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def RelativeName(ref):
    """Converts a resource reference into its relative name string."""
    return ref.RelativeName() if ref else None