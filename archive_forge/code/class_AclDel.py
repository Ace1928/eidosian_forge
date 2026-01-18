from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
class AclDel(object):
    """Represents a logical change from an access control list."""
    scope_regexes = {'All(Users)?$': 'AllUsers', 'AllAuth(enticatedUsers)?$': 'AllAuthenticatedUsers'}

    def __init__(self, identifier):
        self.raw_descriptor = '-d {0}'.format(identifier)
        self.identifier = identifier
        for regex, scope in self.scope_regexes.items():
            if re.match(regex, self.identifier, re.IGNORECASE):
                self.identifier = scope
        self.scope_type = 'Any'
        self.perm = 'NONE'

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

    def Execute(self, storage_url, current_acl, command_name, logger):
        logger.debug('Executing %s %s on %s', command_name, self.raw_descriptor, storage_url)
        matching_entries = list(self._YieldMatchingEntries(current_acl))
        for entry in matching_entries:
            current_acl.remove(entry)
        logger.debug('New Acl:\n%s', str(current_acl))
        return len(matching_entries)