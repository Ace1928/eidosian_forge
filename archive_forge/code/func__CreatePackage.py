from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreatePackage(messages, package_name, package_state, enable_autoupgrade):
    """Creates package in guest policy.

  Args:
    messages: os config guest policy API messages.
    package_name: package name.
    package_state: package states.
    enable_autoupgrade: True or False.

  Returns:
    package in guest policy.
  """
    states = messages.Package.DesiredStateValueValuesEnum
    desired_state = None
    if package_state is agent_policy.OpsAgentPolicy.AgentRule.PackageState.INSTALLED:
        if enable_autoupgrade:
            desired_state = states.UPDATED
        else:
            desired_state = states.INSTALLED
    elif package_state is agent_policy.OpsAgentPolicy.AgentRule.PackageState.REMOVED:
        desired_state = states.REMOVED
    return messages.Package(name=package_name, desiredState=desired_state)