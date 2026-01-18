from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
def _TotalSeconds(delta):
    """Re-implementation of datetime.timedelta.total_seconds() for Python 2.6."""
    return delta.days * 24 * 60 * 60 + delta.seconds