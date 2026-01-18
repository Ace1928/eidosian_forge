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
def _ParseHttpRequestArgs(args, task_type, messages):
    """Parses the attributes of 'args' for Task.httpRequest."""
    if task_type == constants.HTTP_TASK:
        http_method = messages.HttpRequest.HttpMethodValueValuesEnum(args.method.upper()) if args.IsSpecified('method') else None
        return messages.HttpRequest(headers=_ParseHeaderArg(args, messages.HttpRequest.HeadersValue), httpMethod=http_method, body=_ParseBodyArgs(args), url=args.url, oauthToken=_ParseOAuthArgs(args, messages), oidcToken=_ParseOidcArgs(args, messages))