from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupPolicyEnabledArg(parser):
    """Adds a --enabled arg to the given parser."""
    parser.add_argument('--enabled', type=arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), help='The Boolean value indiciating whether backups are made automatically according to the schedules.\n      If enabled, this will be applied to all volumes that have this backup policy attached and enforced on\n      the volume level. If not specified, the default is true.')