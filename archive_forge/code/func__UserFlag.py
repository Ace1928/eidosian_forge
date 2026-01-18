from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def _UserFlag():
    """Returns a flag for setting an auth user."""
    return base.Argument('--auth-user', help="      The test user to log in as. Required if `--auth-type` is not 'none'.\n      'google' login requires a full email address. Cannot be your own account.\n      ")