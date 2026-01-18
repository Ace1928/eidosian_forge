from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.command_lib.dns import resource_args
from googlecloudsdk.command_lib.dns import util as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _AddArgsCommon(parser):
    """Adds the common arguments for all versions."""
    flags.GetLocalDataResourceRecordSets().AddToParser(parser)
    flags.GetResponsePolicyRulesBehavior().AddToParser(parser)
    flags.GetLocationArg().AddToParser(parser)
    parser.add_argument('--dns-name', required=False, help='DNS name (wildcard or exact) to apply this rule to.')