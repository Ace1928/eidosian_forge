from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InsufficientPermissionsError(exceptions.Error):
    """An error raised when the caller does not have sufficient permissions."""

    def __init__(self):
        message = "Caller doesn't have sufficient permissions."
        super(InsufficientPermissionsError, self).__init__(message)