from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def UpdateHTTPMaskHook(unused_ref, args, req):
    """Constructs updateMask for patch requests of PubSub targets.

  Args:
    unused_ref: A resource ref to the parsed Job resource
    args: The parsed args namespace from CLI
    req: Created Patch request for the API call.

  Returns:
    Modified request for the API call.
  """
    http_fields = {'--message-body': 'httpTarget.body', '--message-body-from-file': 'httpTarget.body', '--uri': 'httpTarget.uri', '--http-method': 'httpTarget.httpMethod', '--clear-headers': 'httpTarget.headers', '--remove-headers': 'httpTarget.headers', '--update-headers': 'httpTarget.headers', '--oidc-service-account-email': 'httpTarget.oidcToken.serviceAccountEmail', '--oidc-token-audience': 'httpTarget.oidcToken.audience', '--oauth-service-account-email': 'httpTarget.oauthToken.serviceAccountEmail', '--oauth-token-scope': 'httpTarget.oauthToken.scope', '--clear-auth-token': 'httpTarget.oidcToken.serviceAccountEmail,httpTarget.oidcToken.audience,httpTarget.oauthToken.serviceAccountEmail,httpTarget.oauthToken.scope'}
    req.updateMask = _GenerateUpdateMask(args, http_fields)
    return req