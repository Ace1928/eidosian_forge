from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import p12_service_account
from googlecloudsdk.core.util import files
from oauth2client import service_account
def CredentialsFromAdcFile(filename):
    """Load credentials from given service account json file."""
    content = files.ReadFileContents(filename)
    try:
        json_key = json.loads(content)
        return CredentialsFromAdcDict(json_key)
    except ValueError as e:
        raise BadCredentialFileException('Could not read json file {0}: {1}'.format(filename, e))