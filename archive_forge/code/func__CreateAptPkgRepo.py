from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateAptPkgRepo(messages, repo_name):
    """Create an apt repo in guest policy.

  Args:
    messages: os config guest policy api messages.
    repo_name: repository name.

  Returns:
    An apt repo in guest policy.
  """
    return messages.PackageRepository(apt=messages.AptRepository(uri='http://packages.cloud.google.com/apt', distribution=repo_name, components=['main'], gpgKey='https://packages.cloud.google.com/apt/doc/apt-key.gpg'))