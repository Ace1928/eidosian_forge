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
def _RegisterGKE(self, gke_cluster_resource_link, gke_cluster_uri, project, location, args):
    """Register a GKE cluster without installing Connect agent."""
    obj = None
    issuer_url = None
    if args.enable_workload_identity:
        issuer_url = gke_cluster_uri
    try:
        obj = api_util.CreateMembership(project=project, membership_id=args.MEMBERSHIP_NAME, description=args.MEMBERSHIP_NAME, location=location, gke_cluster_self_link=gke_cluster_resource_link, external_id=None, release_track=self.ReleaseTrack(), issuer_url=issuer_url, oidc_jwks=None, api_server_version=None)
    except apitools_exceptions.HttpConflictError as e:
        error = core_api_exceptions.HttpErrorPayload(e)
        if error.status_description != 'ALREADY_EXISTS':
            raise
        resource_name = api_util.MembershipRef(project, location, args.MEMBERSHIP_NAME)
        obj = api_util.GetMembership(resource_name, self.ReleaseTrack())
        if obj.endpoint.gkeCluster.resourceLink == gke_cluster_resource_link:
            log.status.Print('Membership [{}] already registered with the cluster [{}] in the Fleet.'.format(resource_name, obj.endpoint.gkeCluster.resourceLink))
        else:
            raise exceptions.Error('membership [{}] already exists in the Fleet with another cluster link [{}]. If this operation is intended, please delete the membership and register again.'.format(resource_name, obj.endpoint.gkeCluster.resourceLink))
    log.status.Print('Finished registering to the Fleet.')
    return obj