from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import store as c_store
from oauth2client import client
from google.auth import exceptions as google_auth_exceptions
class LoadingCredentialsError(exceptions.Error):
    """Reraise on oauth2client and google-auth errors."""