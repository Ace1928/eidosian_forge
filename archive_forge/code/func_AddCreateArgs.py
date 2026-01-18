from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import arg_parsers
def AddCreateArgs(parser):
    """Add arguments for the Create command.

  Args:
    parser: A given parser.
  """
    _AddGroupLabelsArgument(parser)
    _AddInstancesArgument(parser)
    _AddZonesArgument(parser)