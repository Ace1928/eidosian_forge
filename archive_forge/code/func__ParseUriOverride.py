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
def _ParseUriOverride(messages, scheme=None, host=None, port=None, path=None, query=None, mode=None):
    """Parses the attributes of 'args' for URI Override."""
    scheme = messages.UriOverride.SchemeValueValuesEnum(scheme.upper()) if scheme else None
    port = int(port) if port else None
    uri_override_enforce_mode = messages.UriOverride.UriOverrideEnforceModeValueValuesEnum(mode.upper()) if mode else None
    return messages.UriOverride(scheme=scheme, host=host, port=port, pathOverride=messages.PathOverride(path=path), queryOverride=messages.QueryOverride(queryParams=query), uriOverrideEnforceMode=uri_override_enforce_mode)