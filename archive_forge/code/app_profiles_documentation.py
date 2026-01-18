from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import exceptions
Update an app profile.

  Args:
    app_profile_ref: A resource reference of the app profile to update.
    cluster: string, The cluster id for the app profile to route to using single
      cluster routing.
    description: string, A description of the app profile.
    multi_cluster: bool, Whether this app profile should route to multiple
      clusters, instead of single cluster.
    restrict_to: list[string] The list of cluster IDs for the app profile to
      route to using multi cluster routing.
    failover_radius: string, Restricts clusters that requests can fail over to
      by proximity with multi cluster routing.
    transactional_writes: bool, Whether this app profile has transactional
      writes enabled. This is only possible when using single cluster routing.
    row_affinity: bool, Whether to use row affinity sticky routing. If None,
      then no change should be made.
    priority: string, The request priority of the new app profile.
    data_boost: bool, If the app profile should use Standard Isolation.
    data_boost_compute_billing_owner: string, The compute billing owner for Data
      Boost.
    force: bool, Whether to ignore API warnings and create forcibly.

  Raises:
    ConflictingArgumentsException,
    OneOfArgumentsRequiredException:
      See _AppProfileChecks(...)

  Returns:
    Long running operation.
  