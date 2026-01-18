from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
def _AddUpstreamFleetFlag(self, group: parser_arguments.ArgumentInterceptor):
    """Adds the --upstream-fleet flag.

    Args:
      group: The group that should contain the flag.
    """
    group.add_argument('--upstream-fleet', type=str, help='        The upstream fleet. GKE will finish upgrades on the upstream fleet\n        before applying the same upgrades to the current fleet.\n\n        To configure the upstream fleet, run:\n\n        $ {command}             --upstream-fleet={upstream_fleet}\n        ')