from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def _ParseHttpTargetArgs(args, queue_type, messages):
    """Parses the attributes of 'args' for Queue.HttpTarget."""
    if queue_type == constants.PUSH_QUEUE:
        uri_override = _ParseHttpRoutingOverrideArgs(args, messages)
        http_method = messages.HttpTarget.HttpMethodValueValuesEnum(args.http_method_override.upper()) if args.IsSpecified('http_method_override') else None
        oauth_token = _ParseHttpTargetOAuthArgs(args, messages)
        oidc_token = _ParseHttpTargetOidcArgs(args, messages)
        if uri_override is None and http_method is None and (oauth_token is None) and (oidc_token is None):
            return None
        return messages.HttpTarget(uriOverride=uri_override, headerOverrides=_ParseHttpTargetHeaderArg(args, messages), httpMethod=http_method, oauthToken=oauth_token, oidcToken=oidc_token)