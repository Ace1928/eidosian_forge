from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _ClassifyScopeIdentifier(text):
    re_map = {'AllAuthenticatedUsers': '^(AllAuthenticatedUsers|AllAuth)$', 'AllUsers': '^(AllUsers|All)$', 'Email': '^.+@.+\\..+$', 'Id': '^[0-9A-Fa-f]{64}$', 'Domain': '^[^@]+\\.[^@]+$', 'Project': '(owners|editors|viewers)\\-.+$'}
    for type_string, regex in re_map.items():
        if re.match(regex, text, re.IGNORECASE):
            return type_string