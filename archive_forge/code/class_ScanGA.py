from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.ondemandscanning import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import ondemandscanning_util as ods_util
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import platforms
import six
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ScanGA(ScanBeta):
    """Perform a vulnerability scan on a container image.

  You can scan a container image in a Google Cloud registry (Artifact Registry
  or Container Registry), or a local container image.

  Reference an image by tag or digest using any of the formats:

    Artifact Registry:
      LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE[:tag]
      LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:digest

    Container Registry:
      [LOCATION.]gcr.io/PROJECT-ID/REPOSITORY-ID/IMAGE[:tag]
      [LOCATION.]gcr.io/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:digest

    Local:
      IMAGE[:tag]
  """

    def AnalyzePackages(self, args, pkgs):
        return api_util.AnalyzePackagesGA(properties.VALUES.core.project.Get(required=True), args.location, args.RESOURCE_URI, pkgs)

    def GetMessages(self):
        return api_util.GetMessages('v1')

    def WaitForOperation(self, op):
        return ods_util.WaitForOperation(op, 'v1')