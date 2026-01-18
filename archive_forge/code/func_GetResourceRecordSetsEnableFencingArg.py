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
def GetResourceRecordSetsEnableFencingArg(required=False):
    """Returns --enable-geo-fencing command line arg value."""
    return base.Argument('--enable-geo-fencing', action='store_true', required=required, help='Specifies whether to enable fencing for geo queries.')