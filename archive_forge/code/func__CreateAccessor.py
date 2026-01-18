from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six
@staticmethod
def _CreateAccessor(i):
    """Create an tuple accessor property."""
    return property(lambda tpl: tpl[i])