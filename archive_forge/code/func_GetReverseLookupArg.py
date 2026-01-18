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
def GetReverseLookupArg():
    return base.Argument('--managed-reverse-lookup', action='store_true', default=None, help='Specifies whether this zone is a managed reverse lookup zone, required for Cloud DNS to correctly resolve Non-RFC1918 PTR records.')