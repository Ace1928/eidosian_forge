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
def GetPolicyAltNameServersArg():
    return base.Argument('--alternative-name-servers', type=arg_parsers.ArgList(), metavar='NAME_SERVERS', help='List of alternative name servers to forward to. Non-RFC1918 addresses will forward to the target through the Internet.RFC1918 addresses will forward through the VPC.')