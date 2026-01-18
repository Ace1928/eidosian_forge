from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def GenerateExclusivityManifest(crd_manifest, cr_manifest, membership_ref, release_track=None):
    """Generate the CR(D) manifests to apply to the registered cluster.

  Args:
    crd_manifest: the YAML manifest of the Membership CRD fetched from the
      cluster.
    cr_manifest: the YAML manifest of the Membership CR fetched from the
      cluster.
    membership_ref: the full resource name of the membership.
    release_track: the release_track used in the gcloud command, or None if it
      is not available.

  Returns:
    the GenerateExclusivityManifestResponse from API.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error.
  """
    release_track = calliope_base.ReleaseTrack.BETA
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    return client.projects_locations_memberships.GenerateExclusivityManifest(client.MESSAGES_MODULE.GkehubProjectsLocationsMembershipsGenerateExclusivityManifestRequest(name=membership_ref, crdManifest=crd_manifest, crManifest=cr_manifest))