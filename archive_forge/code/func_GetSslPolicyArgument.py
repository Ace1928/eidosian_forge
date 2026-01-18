from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetSslPolicyArgument(required=True, plural=False):
    """Returns the resource argument object for the SSL policy flag."""
    return compute_flags.ResourceArgument(name='SSL_POLICY', resource_name='SSL policy', completer=SslPoliciesCompleter, plural=plural, custom_plural='SSL policies', required=required, global_collection='compute.sslPolicies')