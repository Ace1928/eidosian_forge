from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateZypperPkgRepo(messages, repo_id, display_name, repo_name):
    """Create a zypper repo in guest policy.

  Args:
    messages: os config guest policy api messages.
    repo_id: 'google-cloud-logging' or 'google-cloud-monitoring'.
    display_name: 'Google Cloud Logging Agent Repository' or 'Google Cloud
      Monitoring Agent Repository'.
    repo_name: repository name.

  Returns:
    zypper repos in guest policy.
  """
    return messages.PackageRepository(zypper=messages.ZypperRepository(id=repo_id, displayName=display_name, baseUrl='https://packages.cloud.google.com/yum/repos/%s' % repo_name, gpgKeys=['https://packages.cloud.google.com/yum/doc/yum-key.gpg', 'https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg']))