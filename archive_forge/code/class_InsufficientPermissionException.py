from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InsufficientPermissionException(exceptions.Error):
    """Indicates that a user is missing required permissions for an operation."""

    def __init__(self, resource, missing_permissions):
        """Create a new InsufficientPermissionException.

    Args:
      resource: str, The resource on which the user needs permissions.
      missing_permissions: iterable, The missing permissions.
    """
        super(InsufficientPermissionException, self).__init__('The current user does not have permissions for this operation. Please ensure you have {} permissions on the {} and that you are logged-in as the correct user and try again.'.format(','.join(missing_permissions), resource))