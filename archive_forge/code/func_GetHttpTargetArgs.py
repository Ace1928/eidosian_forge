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
def GetHttpTargetArgs(queue_config):
    """Returns a pair of each http target attribute and its value in the queue."""
    http_uri_override = queue_config.httpTarget.uriOverride if queue_config.httpTarget is not None else None
    http_method_override = queue_config.httpTarget.httpMethod if queue_config.httpTarget is not None else None
    http_header_override = queue_config.httpTarget.headerOverrides if queue_config.httpTarget is not None else None
    http_oauth_email_override = queue_config.httpTarget.oauthToken.serviceAccountEmail if queue_config.httpTarget is not None and queue_config.httpTarget.oauthToken is not None else None
    http_oauth_scope_override = queue_config.httpTarget.oauthToken.scope if queue_config.httpTarget is not None and queue_config.httpTarget.oauthToken is not None else None
    http_oidc_email_override = queue_config.httpTarget.oidcToken.serviceAccountEmail if queue_config.httpTarget is not None and queue_config.httpTarget.oidcToken is not None else None
    http_oidc_audience_override = queue_config.httpTarget.oidcToken.audience if queue_config.httpTarget is not None and queue_config.httpTarget.oidcToken is not None else None
    return {'http_uri_override': http_uri_override, 'http_method_override': http_method_override, 'http_header_override': http_header_override, 'http_oauth_email_override': http_oauth_email_override, 'http_oauth_scope_override': http_oauth_scope_override, 'http_oidc_email_override': http_oidc_email_override, 'http_oidc_audience_override': http_oidc_audience_override}