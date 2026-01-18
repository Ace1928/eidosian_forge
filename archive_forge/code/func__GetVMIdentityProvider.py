from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _GetVMIdentityProvider(membership_manifest, workload_namespace):
    """Get the identity provider for the VMs.

  Args:
    membership_manifest: The membership manifest from the cluster.
    workload_namespace: The namespace of the VM workload.

  Returns:
    The identity provider value to be used on the VM connected to the cluster.

  Raises:
    ClusterError: If the membership manifest cannot be read.
  """
    if not membership_manifest:
        raise ClusterError('Cannot verify an empty membership from the cluster')
    try:
        membership_data = yaml.load(membership_manifest)
    except yaml.Error as e:
        raise exceptions.Error('Invalid membership from the cluster {}'.format(membership_manifest), e)
    owner_id = _GetNestedKeyFromManifest(membership_data, 'spec', 'owner', 'id')
    if not owner_id:
        raise ClusterError('Invalid membership does not have an owner id. Please make sure your cluster is correctly registered and retry.')
    membership_name = _ParseMembershipName(owner_id)
    membership = api_util.GetMembership(membership_name)
    if not membership.uniqueId:
        raise exceptions.Error('Invalid membership {} does not have a unique_Id field. Please make sure your cluster is correctly registered and retry.'.format(membership_name))
    return '{}@google@{}'.format(membership.uniqueId, workload_namespace)