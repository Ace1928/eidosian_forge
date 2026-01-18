from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import encoding
import six
def _HasDefaultRepr(obj):
    """Returns True if obj has default __repr__ and __str__ methods."""
    try:
        d = obj.__class__.__dict__
        return '__str__' not in d and '__repr__' not in d
    except AttributeError:
        return False