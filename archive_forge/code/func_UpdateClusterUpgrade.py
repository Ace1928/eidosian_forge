from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container.fleet.scopes.rollout_sequencing import base
from googlecloudsdk.core import log
def UpdateClusterUpgrade(response, args):
    """Updates Cluster Upgrade feature.

  Args:
    response: reference to the Scope object.
    args: command line arguments.

  Returns:
    response
  """
    update_cmd = base.UpdateCommand(args)
    if update_cmd.IsClusterUpgradeRequest():
        if args.IsKnownAndSpecified('async_') and args.async_:
            msg = 'Both --async and Rollout Sequencing flag(s) specified. Cannot modify cluster upgrade feature until scope operation has completed. Ignoring Rollout Sequencing flag(s). Use synchronous update command to apply desired cluster upgrade feature changes to the current scope.'
            log.warning(msg)
            return response
        enable_cmd = base.EnableCommand(args)
        feature = enable_cmd.GetWithForceEnable()
        scope_name = base.ClusterUpgradeCommand.GetScopeNameWithProjectNumber(response.name)
        updated_feature = update_cmd.Update(feature, scope_name)
        describe_cmd = base.DescribeCommand(args)
        return describe_cmd.AddClusterUpgradeInfoToScope(response, scope_name, updated_feature)
    return response