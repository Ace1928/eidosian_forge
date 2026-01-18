from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def GetUsageMetavar(self, *args, **kwargs):
    """Forwards default usage metavar for arg_type."""
    if self._is_usage_type:
        return self.arg_type.GetUsageMetavar(*args, **kwargs)
    else:
        return None