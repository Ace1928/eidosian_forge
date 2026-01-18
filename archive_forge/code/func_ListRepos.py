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
def ListRepos(project_id, status, namespace, membership, selector, targets):
    """List repos across clusters.

  Args:
    project_id: project id that the command should list the repo from.
    status: status of the repo that the list result should contain
    namespace: namespace of the repo that the command should list.
    membership: membership name that the repo should be from.
    selector: label selectors for repo. It applies to the RootSync|RepoSync CRs.
    targets: The targets from which to list the repos. The value should be one
      of "all", "fleet-clusters" and "config-controller".

  Returns:
    A list of RepoStatus.

  """
    if targets and targets not in ['all', 'fleet-clusters', 'config-controller']:
        raise exceptions.ConfigSyncError('--targets must be one of "all", "fleet-clusters" and "config-controller"')
    if targets != 'fleet-clusters' and membership:
        raise exceptions.ConfigSyncError('--membership should only be specified when --targets=fleet-clusters')
    if status not in ['all', 'synced', 'error', 'pending', 'stalled']:
        raise exceptions.ConfigSyncError('--status must be one of "all", "synced", "pending", "error", "stalled"')
    selector_map, err = _ParseSelector(selector)
    if err:
        raise exceptions.ConfigSyncError(err)
    repo_cross_clusters = RawRepos()
    if targets == 'all' or targets == 'config-controller':
        clusters = []
        try:
            clusters = utils.ListConfigControllerClusters(project_id)
        except exceptions.ConfigSyncError as err:
            log.error(err)
        if clusters:
            for cluster in clusters:
                try:
                    utils.KubeconfigForCluster(project_id, cluster[1], cluster[0])
                    _AppendReposFromCluster(cluster[0], repo_cross_clusters, 'Config Controller', namespace, selector_map)
                except exceptions.ConfigSyncError as err:
                    log.error(err)
    if targets == 'all' or targets == 'fleet-clusters':
        try:
            memberships = utils.ListMemberships(project_id)
        except exceptions.ConfigSyncError as err:
            raise err
        for member in memberships:
            if not utils.MembershipMatched(member, membership):
                continue
            try:
                utils.KubeconfigForMembership(project_id, member)
                _AppendReposFromCluster(member, repo_cross_clusters, 'Membership', namespace, selector_map)
            except exceptions.ConfigSyncError as err:
                log.error(err)
    return _AggregateRepoStatus(repo_cross_clusters, status)