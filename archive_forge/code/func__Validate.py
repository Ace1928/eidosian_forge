from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _Validate(self):
    """Validates a parsed AclChange object."""

    def _ThrowError(msg):
        raise CommandException('{0} is not a valid ACL change\n{1}'.format(self.raw_descriptor, msg))
    if self.scope_type not in self.scope_types:
        _ThrowError('{0} is not a valid scope type'.format(self.scope_type))
    if self.scope_type in self.public_scopes and self.identifier:
        _ThrowError('{0} requires no arguments'.format(self.scope_type))
    if self.scope_type in self.id_scopes and (not self.identifier):
        _ThrowError('{0} requires an id'.format(self.scope_type))
    if self.scope_type in self.email_scopes and (not self.identifier):
        _ThrowError('{0} requires an email address'.format(self.scope_type))
    if self.scope_type in self.domain_scopes and (not self.identifier):
        _ThrowError('{0} requires domain'.format(self.scope_type))
    if self.perm not in self.permission_shorthand_mapping.values():
        perms = ', '.join(set(self.permission_shorthand_mapping.values()))
        _ThrowError('Allowed permissions are {0}'.format(perms))