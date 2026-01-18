from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _AggregateRepoStatus(repos_cross_clusters, status):
    """Aggregate the repo status from multiple clusters.

  Args:
    repos_cross_clusters: The repos read from multiple clusters.
    status: The status used for filtering the list results.

  Returns:
    The list of RepoStatus after aggregation.
  """
    repos = []
    for git, rs in repos_cross_clusters.GetRepos().items():
        repo_status = _GetRepoStatus(rs, git)
        if not _StatusMatched(status, repo_status):
            continue
        repos.append(repo_status)
    return repos