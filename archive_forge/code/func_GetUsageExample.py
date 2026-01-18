from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def GetUsageExample(self, *args, **kwargs):
    """Forwards default usage example for arg_type."""
    if self._is_usage_type:
        return self.arg_type.GetUsageExample(*args, **kwargs)
    else:
        return None