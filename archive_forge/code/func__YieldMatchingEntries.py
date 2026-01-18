from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _YieldMatchingEntries(self, current_acl):
    """Generator that yields entries that match the change descriptor.

    Args:
      current_acl: An instance of apitools_messages.BucketAccessControls or
                   ObjectAccessControls which will be searched for matching
                   entries.

    Yields:
      An apitools_messages.BucketAccessControl or ObjectAccessControl.
    """
    for entry in current_acl:
        if entry.entityId and self.identifier.lower() == entry.entityId.lower():
            yield entry
        elif entry.email and self.identifier.lower() == entry.email.lower():
            yield entry
        elif entry.domain and self.identifier.lower() == entry.domain.lower():
            yield entry
        elif entry.projectTeam and self.identifier.lower() == '%s-%s'.lower() % (entry.projectTeam.team, entry.projectTeam.projectNumber):
            yield entry
        elif entry.entity.lower() == 'allusers' and self.identifier == 'AllUsers':
            yield entry
        elif entry.entity.lower() == 'allauthenticatedusers' and self.identifier == 'AllAuthenticatedUsers':
            yield entry