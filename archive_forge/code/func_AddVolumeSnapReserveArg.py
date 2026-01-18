from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeSnapReserveArg(parser):
    """Adds the --snap-reserve arg to the arg parser."""
    action = actions.DeprecationAction('snap-reserve', warn='The {flag_name} option is deprecated', removed=False)
    parser.add_argument('--snap-reserve', type=float, help='The percentage of volume storage reserved for snapshot storage.\n      The default value for this is 0 percent', action=action)