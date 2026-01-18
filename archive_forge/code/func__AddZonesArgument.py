from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import arg_parsers
def _AddZonesArgument(parser):
    parser.add_argument('--zones', metavar='ZONE_NAME', type=arg_parsers.ArgList(), help="      A list of zones to filter instances to apply the policy.\n\n      To list available zones, run:\n\n        $ gcloud compute zones list\n\n      The use of the ``--zones'' and ``--group-labels'' flags is recommended for\n      production environments. For testing and development, it's more common to\n      select instances directly via the ``--instances'' flag.\n      ")