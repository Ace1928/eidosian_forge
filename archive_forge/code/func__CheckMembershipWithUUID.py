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
def _CheckMembershipWithUUID(self, resource_name, membership_name):
    """Checks for an existing Membership with UUID.

    In the past, by default we used Cluster UUID to create a membership. Now
    we use user supplied membership_name. This check ensures that we don't
    reregister a cluster.

    Args:
      resource_name: The full membership resource name using the cluster uuid.
      membership_name: User supplied membership_name.

    Returns:
     The Membership resource or None.

    Raises:
      exceptions.Error: If it fails to getMembership.
    """
    obj = None
    try:
        obj = api_util.GetMembership(resource_name, self.ReleaseTrack())
        if hasattr(obj, 'description') and obj.description != membership_name:
            raise exceptions.Error('There is an existing membership, [{}], that conflicts with [{}]. Please delete it before continuing:\n\n  gcloud {}container fleet memberships delete {}'.format(obj.description, membership_name, hub_util.ReleaseTrackCommandPrefix(self.ReleaseTrack()), resource_name))
    except apitools_exceptions.HttpNotFoundError:
        pass
    return obj