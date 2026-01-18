from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def _UrlFlag():
    """Returns a flag for setting an auth url."""
    return base.Argument('--auth-url', help="      URL of the login page for your site. Required if `--auth-type` is\n      'custom'. Otherwise, it should not be set.\n      ")