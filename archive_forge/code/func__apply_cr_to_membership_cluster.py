from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _apply_cr_to_membership_cluster(kube_client, yaml_string):
    """Apply the CloudRun custom resource to the cluster.

  Args:
    kube_client: A Kubernetes client.
    yaml_string: the CloudRun YAML file.
  """
    _, err = kube_client.Apply(yaml_string)
    if err:
        raise exceptions.Error('Failed to apply manifest to cluster: {}'.format(err))