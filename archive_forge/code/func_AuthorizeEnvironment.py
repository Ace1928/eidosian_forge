from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
def AuthorizeEnvironment():
    """Pushes gcloud command-line tool credentials to the user's environment."""
    client = apis.GetClientInstance('cloudshell', 'v1')
    messages = apis.GetMessagesModule('cloudshell', 'v1')
    access_token = store.GetFreshAccessTokenIfEnabled(min_expiry_duration=MIN_CREDS_EXPIRY_SECONDS)
    if access_token:
        client.users_environments.Authorize(messages.CloudshellUsersEnvironmentsAuthorizeRequest(name=DEFAULT_ENVIRONMENT_NAME, authorizeEnvironmentRequest=messages.AuthorizeEnvironmentRequest(accessToken=access_token)))