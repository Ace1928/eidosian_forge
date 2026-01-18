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
def _AppendReposFromCluster(membership, repos_cross_clusters, cluster_type, namespaces, selector):
    """List all the RepoSync and RootSync CRs from the given cluster.

  Args:
    membership: The membership name or cluster name of the current cluster.
    repos_cross_clusters: The repos across multiple clusters.
    cluster_type: The type of the current cluster. It is either a Fleet-cluster
      or a Config-controller cluster.
    namespaces: The namespaces that the list should get RepoSync|RootSync from.
    selector: The label selector that the RepoSync|RootSync should match.

  Returns:
    None

  Raises:
    Error: errors that happen when listing the CRs from the cluster.
  """
    utils.GetConfigManagement(membership)
    params = []
    if not namespaces or '*' in namespaces:
        params = [['--all-namespaces']]
    else:
        params = [['-n', ns] for ns in namespaces.split(',')]
    all_repos = []
    errors = []
    for p in params:
        repos, err = utils.RunKubectl(['get', 'rootsync,reposync', '-o', 'json'] + p)
        if err:
            errors.append(err)
            continue
        if repos:
            obj = json.loads(repos)
            if 'items' in obj:
                if namespaces and '*' in namespaces:
                    for item in obj['items']:
                        ns = _GetPathValue(item, ['metadata', 'namespace'], '')
                        if fnmatch.fnmatch(ns, namespaces):
                            all_repos.append(item)
                else:
                    all_repos += obj['items']
    if errors:
        raise exceptions.ConfigSyncError('Error getting RootSync and RepoSync custom resources: {}'.format(errors))
    count = 0
    for repo in all_repos:
        if not _LabelMatched(repo, selector):
            continue
        repos_cross_clusters.AddRepo(membership, repo, None, cluster_type)
        count += 1
    if count > 0:
        log.status.Print('getting {} RepoSync and RootSync from {}'.format(count, membership))