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
def DescribeRepo(project, name, namespace, source, repo_cluster, managed_resources):
    """Describe a repo for the detailed status and managed resources.

  Args:
    project: The project id the repo is from.
    name: The name of the correspoinding RepoSync|RootSync CR.
    namespace: The namespace of the correspoinding RepoSync|RootSync CR.
    source: The source of the repo.
    repo_cluster: The cluster that the repo is synced to.
    managed_resources: The status to filter the managed resources for the
      output.

  Returns:
    It returns an instance of DescribeResult

  """
    if name and source or (namespace and source):
        raise exceptions.ConfigSyncError('--sync-name and --sync-namespace cannot be specified together with --source.')
    if name and (not namespace) or (namespace and (not name)):
        raise exceptions.ConfigSyncError('--sync-name and --sync-namespace must be specified together.')
    if managed_resources not in ['all', 'current', 'inprogress', 'notfound', 'failed', 'unknown']:
        raise exceptions.ConfigSyncError('--managed-resources must be one of all, current, inprogress, notfound, failed or unknown')
    repo_cross_clusters = RawRepos()
    clusters = []
    try:
        clusters = utils.ListConfigControllerClusters(project)
    except exceptions.ConfigSyncError as err:
        log.error(err)
    if clusters:
        for cluster in clusters:
            if repo_cluster and repo_cluster != cluster[0]:
                continue
            try:
                utils.KubeconfigForCluster(project, cluster[1], cluster[0])
                _AppendReposAndResourceGroups(cluster[0], repo_cross_clusters, 'Config Controller', name, namespace, source)
            except exceptions.ConfigSyncError as err:
                log.error(err)
    try:
        memberships = utils.ListMemberships(project)
    except exceptions.ConfigSyncError as err:
        raise err
    for membership in memberships:
        if repo_cluster and repo_cluster != membership:
            continue
        try:
            utils.KubeconfigForMembership(project, membership)
            _AppendReposAndResourceGroups(membership, repo_cross_clusters, 'Membership', name, namespace, source)
        except exceptions.ConfigSyncError as err:
            log.error(err)
    repo = _Describe(managed_resources, repo_cross_clusters)
    return repo