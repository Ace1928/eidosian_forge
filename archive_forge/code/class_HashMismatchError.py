from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
class HashMismatchError(Error):
    """Error raised when hashes don't match after operation."""