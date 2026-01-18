from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import arg_parsers
def _AddInstancesArgument(parser):
    parser.add_argument('--instances', metavar='zones/ZONE_NAME/instances/INSTANCE_NAME', type=arg_parsers.ArgList(), help="      A list of fully-qualified names to filter instances that the policy\n      applies to.\n\n      Each item in the list must be in the format of\n      `zones/ZONE_NAME/instances/INSTANCE_NAME`. The policy can also target\n      instances that are not yet created.\n\n      To list all existing instances, run:\n\n        $ gcloud compute instances list\n\n      The ``--instances'' flag is recommended for use during development and\n      testing. In production environments, it's more common to select instances\n      via a combination of ``--zones'' and ``--group-labels''.\n      ")