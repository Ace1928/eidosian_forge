from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSubsettingSubsetSize(parser):
    parser.add_argument('--subsetting-subset-size', type=int, help='      Number of backends per backend group assigned to each proxy instance\n      or each service mesh client. Can only be set if subsetting policy is\n      CONSISTENT_HASH_SUBSETTING and load balancing scheme is either\n      INTERNAL_MANAGED or INTERNAL_SELF_MANAGED.\n      ')