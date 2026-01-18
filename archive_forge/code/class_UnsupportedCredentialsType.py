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
class UnsupportedCredentialsType(Error):
    """Raised when given type credentials cannot be created."""