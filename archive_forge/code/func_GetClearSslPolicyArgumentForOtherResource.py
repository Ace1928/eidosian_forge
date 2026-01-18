from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetClearSslPolicyArgumentForOtherResource(proxy_type, required=False):
    """Returns the flag for clearing the SSL policy."""
    return base.Argument('--clear-ssl-policy', action='store_true', default=False, required=required, help='      Removes any attached SSL policy from the {} proxy.\n      '.format(proxy_type))