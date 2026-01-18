from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetResponsePolicyNetworksArg(required=False):
    return base.Argument('--networks', type=arg_parsers.ArgList(), required=required, metavar='NETWORKS', help='The comma-separated list of network names to associate with the response policy.')