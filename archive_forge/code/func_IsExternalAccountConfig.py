from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
def IsExternalAccountConfig(content_json):
    """Returns whether a JSON content corresponds to an external account cred."""
    return (content_json or {}).get('type') == _EXTERNAL_ACCOUNT_TYPE