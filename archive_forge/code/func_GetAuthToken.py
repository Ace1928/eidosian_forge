import copy
import json
import os
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core.credentials import store as c_store
def GetAuthToken(account, operation, impersonated=False):
    """Generate a JSON object containing the current gcloud auth token."""
    try:
        access_token = c_store.GetFreshAccessToken(account, allow_account_impersonation=impersonated)
        output = {'auth_token': access_token}
    except Exception as e:
        raise c_except.Error('Error retrieving auth credentials for {operation}: {error}. '.format(operation=operation, error=e))
    return json.dumps(output, sort_keys=True)