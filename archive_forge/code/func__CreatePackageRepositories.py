from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreatePackageRepositories(messages, os_type, agent_rules):
    """Create package repositories in guest policy.

  Args:
    messages: os config guest policy api messages.
    os_type: it contains os_version, os_shortname.
    agent_rules: list of agent rules which contains version, package_state, type
      of {logging,metrics}.

  Returns:
    package repos in guest policy.
  """
    package_repos = None
    if os_type.short_name in _APT_OS:
        package_repos = _CreateAptPkgRepos(messages, _APT_CODENAMES.get(os_type.version), agent_rules)
    elif os_type.short_name in _YUM_OS:
        version = os_type.version.split('.')[0]
        version = version.split('*')[0]
        package_repos = _CreateYumPkgRepos(messages, version, agent_rules)
    elif os_type.short_name in _SUSE_OS:
        version = os_type.version.split('.')[0]
        version = version.split('*')[0]
        package_repos = _CreateZypperPkgRepos(messages, version, agent_rules)
    elif os_type.short_name in _WINDOWS_OS:
        package_repos = _CreateGooPkgRepos(messages, 'windows', agent_rules)
    return package_repos