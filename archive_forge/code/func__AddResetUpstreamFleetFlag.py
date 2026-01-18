from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def _AddResetUpstreamFleetFlag(self, group: parser_arguments.ArgumentInterceptor):
    """Adds the --reset-upstream-fleet flag.

    Args:
      group: The group that should contain the flag.
    """
    group.add_argument('--reset-upstream-fleet', action='store_true', default=None, help='        Clears the relationship between the current fleet and its upstream\n        fleet in the rollout sequence.\n\n        To remove the link between the current fleet and its upstream fleet,\n        run:\n\n          $ {command} --reset-upstream-fleet\n        ')