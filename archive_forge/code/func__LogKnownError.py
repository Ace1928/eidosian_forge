from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
def _LogKnownError(known_exc, command_path, print_error):
    """Logs the error message of the known exception."""
    msg = '({0}) {1}'.format(console_attr.SafeText(command_path), console_attr.SafeText(known_exc))
    if isinstance(known_exc, api_exceptions.HttpException):
        service_use_help = _BuildMissingServiceUsePermissionAdditionalHelp(known_exc)
        auth_scopes_help = _BuildMissingAuthScopesAdditionalHelp(known_exc)
        msg = service_use_help.Extend(msg)
        msg = auth_scopes_help.Extend(msg)
    log.debug(msg, exc_info=sys.exc_info())
    if print_error:
        log.error(msg)