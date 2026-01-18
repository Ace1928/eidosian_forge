from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidExclusionException(Error):
    """Raised if a user tries to exclude incompatible metrics."""