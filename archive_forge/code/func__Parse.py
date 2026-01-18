from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _Parse(self, change_descriptor, scope_type):
    """Parses an ACL Change descriptor."""

    def _ClassifyScopeIdentifier(text):
        re_map = {'AllAuthenticatedUsers': '^(AllAuthenticatedUsers|AllAuth)$', 'AllUsers': '^(AllUsers|All)$', 'Email': '^.+@.+\\..+$', 'Id': '^[0-9A-Fa-f]{64}$', 'Domain': '^[^@]+\\.[^@]+$', 'Project': '(owners|editors|viewers)\\-.+$'}
        for type_string, regex in re_map.items():
            if re.match(regex, text, re.IGNORECASE):
                return type_string
    if change_descriptor.count(':') != 1:
        raise CommandException('{0} is an invalid change description.'.format(change_descriptor))
    scope_string, perm_token = change_descriptor.split(':')
    perm_token = perm_token.upper()
    if perm_token in self.permission_shorthand_mapping:
        self.perm = self.permission_shorthand_mapping[perm_token]
    else:
        self.perm = perm_token
    scope_class = _ClassifyScopeIdentifier(scope_string)
    if scope_class == 'Domain':
        self.scope_type = '{0}ByDomain'.format(scope_type)
        self.identifier = scope_string
    elif scope_class in ('Email', 'Id'):
        self.scope_type = '{0}By{1}'.format(scope_type, scope_class)
        self.identifier = scope_string
    elif scope_class == 'AllAuthenticatedUsers':
        self.scope_type = 'AllAuthenticatedUsers'
    elif scope_class == 'AllUsers':
        self.scope_type = 'AllUsers'
    elif scope_class == 'Project':
        self.scope_type = 'Project'
        self.identifier = scope_string
    else:
        self.scope_type = scope_string