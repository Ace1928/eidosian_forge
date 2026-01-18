from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import urllib.parse
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import requests
def _sign_with_iam(account_email, string_to_sign, delegates):
    """Generates a signature using the IAM sign-blob method.

  Args:
    account_email (str): Email of the service account to sign as.
    string_to_sign (str): String to sign.
    delegates (list[str]|None): The list of service accounts in a delegation
      chain specified in --impersonate-service-account.

  Returns:
    A raw signature for the specified string.
  """
    http_client = transports.GetApitoolsTransport(response_encoding=transport.ENCODING, allow_account_impersonation=False)
    client = apis_internal._GetClientInstance('iamcredentials', 'v1', http_client=http_client)
    messages = client.MESSAGES_MODULE
    response = client.projects_serviceAccounts.SignBlob(messages.IamcredentialsProjectsServiceAccountsSignBlobRequest(name=iam_util.EmailToAccountResourceName(account_email), signBlobRequest=messages.SignBlobRequest(payload=bytes(string_to_sign, 'utf-8'), delegates=[iam_util.EmailToAccountResourceName(delegate) for delegate in delegates or []])))
    return response.signedBlob