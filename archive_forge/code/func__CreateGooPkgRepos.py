from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateGooPkgRepos(messages, repo_distro, agent_rules):
    goo_pkg_repos = []
    for agent_rule in agent_rules:
        template = _AGENT_RULE_TEMPLATES[agent_rule.type]
        repo_name = template.goo_package.repo % (repo_distro, _GetRepoSuffix(agent_rule.version))
        goo_pkg_repos.append(_CreateGooPkgRepo(messages, repo_name))
    return goo_pkg_repos