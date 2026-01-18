from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnsupportedLocationError(exceptions.Error):
    """Raised when the given location is invalid."""