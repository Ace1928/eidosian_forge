from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddCollectorIlbArg(parser, is_for_update=False):
    parser.add_argument('--collector-ilb', required=not is_for_update, help='      Forwarding rule configured as collector. This must be a regional\n      forwarding rule (in the same region) with load balancing scheme INTERNAL\n      and isMirroringCollector set to true.\n\n      You can provide this as the full URL to the forwarding rule, partial URL,\n      or name.\n      For example, the following are valid values:\n        * https://compute.googleapis.com/compute/v1/projects/myproject/\n          regions/us-central1/forwardingRules/fr-1\n        * projects/myproject/regions/us-central1/forwardingRules/fr-1\n        * fr-1\n      ')