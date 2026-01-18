from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.command_lib.container.fleet import agent_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import exclusivity_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util as hub_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _VerifyClusterExclusivity(self, kube_client, parent, membership_id):
    """Verifies that the cluster can be registered to the project.

    Args:
      kube_client: a KubernetesClient
      parent: the parent collection the user is attempting to register the
        cluster with.
      membership_id: the ID of the membership to be created for the cluster.

    Raises:
      apitools.base.py.HttpError: if the API request returns an HTTP error.
      exceptions.Error: if the cluster is in an invalid exclusivity state.
    """
    cr_manifest = ''
    if kube_client.MembershipCRDExists():
        cr_manifest = kube_client.GetMembershipCR()
    res = api_util.ValidateExclusivity(cr_manifest, parent, membership_id, self.ReleaseTrack())
    if res.status.code:
        raise exceptions.Error("Error validating cluster's exclusivity state with the Fleet under parent collection [{}]: {}. Cannot proceed with the cluster registration.".format(parent, res.status.message))