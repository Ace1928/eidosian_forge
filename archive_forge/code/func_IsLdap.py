from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def IsLdap(self):
    """Returns true is the current preferredAuth Method is ldap."""
    try:
        auth_name = self.GetPreferredAuth()
        found_auth = self._FindMatchingAuthMethod(auth_name, 'ldap')
        if found_auth:
            return True
    except (YamlConfigObjectFieldError, KeyError):
        pass
    return False