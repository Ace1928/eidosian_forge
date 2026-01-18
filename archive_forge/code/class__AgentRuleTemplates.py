from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
class _AgentRuleTemplates(collections.namedtuple('_AgentRuleTemplates', ('install_with_version', 'yum_package', 'apt_package', 'zypper_package', 'goo_package', 'run_agent', 'win_run_agent', 'repo_id', 'display_name', 'recipe_name', 'current_major_version'))):
    pass