from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateGooPkgRepo(messages, repo_id):
    """Create a goo repo in guest policy.

  Args:
    messages: os config guest policy api messages.
    repo_id: 'google-cloud-ops-agent-windows-[all|1]'.

  Returns:
    zoo repos in guest policy.
  """
    return messages.PackageRepository(goo=messages.GooRepository(name=repo_id, url='https://packages.cloud.google.com/yuck/repos/%s' % repo_id))