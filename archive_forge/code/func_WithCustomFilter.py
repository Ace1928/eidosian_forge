from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def WithCustomFilter(self, custom_filter):
    """Add a custom filter to this filter."""
    self._custom_filter = custom_filter
    return self