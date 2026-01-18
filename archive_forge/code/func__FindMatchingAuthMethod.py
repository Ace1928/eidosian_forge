from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _FindMatchingAuthMethod(self, method_name, method_type):
    providers = self.GetAuthProviders(name_only=False)
    found = [x for x in providers if x['name'] == method_name and x[method_type] is not None]
    if found:
        return found.pop()
    return None