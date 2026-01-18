from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import exceptions
def _AppProfileChecks(cluster=None, multi_cluster=False, restrict_to=None, failover_radius=None, transactional_writes=None, row_affinity=False, data_boost=False):
    """Create an app profile.

  Args:
    cluster: string, The cluster id for the new app profile to route to using
      single cluster routing.
    multi_cluster: bool, Whether this app profile should route to multiple
      clusters, instead of single cluster.
    restrict_to: list[string] The list of cluster ids for the new app profile to
      route to using multi cluster routing.
    failover_radius: string, Restricts clusters that requests can fail over to
      by proximity with multi cluster routing.
    transactional_writes: bool, Whether this app profile has transactional
      writes enabled. This is only possible when using single cluster routing.
    row_affinity: bool, Whether to use row affinity sticky routing.
    data_boost: bool, If the app profile should use Data Boost Read-only
      Isolation.

  Raises:
    ConflictingArgumentsException:
        If both cluster and multi_cluster are present.
        If both multi_cluster and transactional_writes are present.
        If both cluster and row_affinity are present.
        If both cluster and restrict_to are present.
        If both cluster and failover_radius are present.
        If both multi_cluster and data_boost are present.
        If both transactional_writes and data_boost are present.

    OneOfArgumentsRequiredException: If neither cluster or multi_cluster are
        present.
  """
    if multi_cluster and cluster:
        raise exceptions.ConflictingArgumentsException('--route-to', '--route-any')
    if not multi_cluster and (not cluster):
        raise exceptions.OneOfArgumentsRequiredException(['--route-to', '--route-any'], 'Either --route-to or --route-any must be specified.')
    if multi_cluster and transactional_writes:
        raise exceptions.ConflictingArgumentsException('--route-any', '--transactional-writes')
    if cluster and row_affinity:
        raise exceptions.ConflictingArgumentsException('--route-to', '--row-affinity')
    if cluster and restrict_to:
        raise exceptions.ConflictingArgumentsException('--route-to', '--restrict-to')
    if cluster and failover_radius:
        raise exceptions.ConflictingArgumentsException('--route-to', '--failover-radius')
    if multi_cluster and data_boost:
        raise exceptions.ConflictingArgumentsException('--route-any', '--data-boost')
    if transactional_writes and data_boost:
        raise exceptions.ConflictingArgumentsException('--transactional-writes', '--data-boost')