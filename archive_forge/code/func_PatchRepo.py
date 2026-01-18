from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def PatchRepo(self, repo, update_mask='pubsubConfigs'):
    """Updates a repo's configuration."""
    req = self.messages.SourcerepoProjectsReposPatchRequest(name=repo.name, updateRepoRequest=self.messages.UpdateRepoRequest(repo=repo, updateMask=update_mask))
    return self._client.projects_repos.Patch(req)