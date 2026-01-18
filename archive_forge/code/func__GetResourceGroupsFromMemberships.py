from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetResourceGroupsFromMemberships(project, name, namespace, repo_cluster, membership):
    """List all ResourceGroup CRs from the provided membership cluster.

  Args:
    project: The project id the repo is from.
    name: The name of the corresponding ResourceGroup CR.
    namespace: The namespace of the corresponding ResourceGroup CR.
    repo_cluster: The cluster that the repo is synced to.
    membership: membership name that the repo should be from.

  Returns:
    List of raw ResourceGroup dicts

  """
    resource_groups = []
    try:
        memberships = utils.ListMemberships(project)
    except exceptions.ConfigSyncError as err:
        raise err
    for member in memberships:
        if membership and (not utils.MembershipMatched(member, membership)):
            continue
        if repo_cluster and repo_cluster != member:
            continue
        try:
            utils.KubeconfigForMembership(project, member)
            member_rg = _GetResourceGroups(member, name, namespace)
            if member_rg:
                resource_groups.extend(member_rg)
        except exceptions.ConfigSyncError as err:
            log.error(err)
    return resource_groups