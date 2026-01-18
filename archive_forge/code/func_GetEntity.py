from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def GetEntity(self):
    """Gets an appropriate entity string for an ACL grant."""
    if self.scope_type in ('UserById', 'UserByEmail'):
        return self.user_entity_prefix + self.identifier
    elif self.scope_type in ('GroupById', 'GroupByEmail'):
        return self.group_entity_prefix + self.identifier
    elif self.scope_type == 'Project':
        return self.project_entity_prefix + self.identifier
    elif self.scope_type == 'GroupByDomain':
        return self.domain_entity_prefix + self.identifier
    elif self.scope_type == 'AllAuthenticatedUsers':
        return self.public_entity_all_auth_users
    elif self.scope_type == 'AllUsers':
        return self.public_entity_all_users
    else:
        raise CommandException('Add entry to ACL got unexpected scope type %s.' % self.scope_type)