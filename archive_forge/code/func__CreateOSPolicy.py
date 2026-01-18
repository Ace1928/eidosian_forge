import os
import pathlib
import string
from typing import Optional
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.instances.ops_agents import cloud_ops_agents_policy as agent_policy
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as osconfig
def _CreateOSPolicy(agents_rule: agent_policy.OpsAgentsPolicy.AgentsRule) -> osconfig.OSPolicy:
    """Creates OS Policy from Ops Agents Rule.

  Args:
    agents_rule: User inputed agents rule.

  Returns:
    osconfig.OSPolicy
  """
    template_path = pathlib.Path(os.path.abspath(__file__)).parent
    is_latest = agents_rule.version == '2.*.*' or agents_rule.version == 'latest'
    installed = agents_rule.package_state == agent_policy.OpsAgentsPolicy.AgentsRule.PackageState.INSTALLED
    if installed:
        if is_latest:
            template_name = 'policy_major_version_install.yaml'
        else:
            template_name = 'policy_pin_to_version_install.yaml'
    else:
        template_name = 'policy_uninstall.yaml'
    agent_version = agents_rule.version if installed and (not is_latest) else _GetRepoSuffix(agents_rule.version)
    template = string.Template(files.ReadFileContents(template_path.joinpath(template_name))).safe_substitute(agent_version=agent_version)
    os_policy = encoding.PyValueToMessage(osconfig.OSPolicy, yaml.load(template))
    os_policy.description = 'AUTO-GENERATED VALUE, DO NOT EDIT! | %s' % agents_rule.ToJson()
    return os_policy