from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def GetImageSummaryMetadata(docker_version):
    """Retrieves build and SBOM metadata for a docker image.

  This function is used only for SLSA build level computation and retrieving
  SBOM locations. If the containeranalysis API is disabled for the project, no
  request will be sent and it returns empty metadata resulting in 'unknown' SLSA
  level.

  Args:
    docker_version: docker info about image and project.

  Returns:
    The build and SBOM metadata for the given image.
  """
    metadata = ContainerAnalysisMetadata()
    ca_enabled = enable_api.IsServiceEnabled(docker_version.project, 'containeranalysis.googleapis.com')
    if not ca_enabled:
        return metadata
    docker_urls = ['https://{}'.format(docker_version.GetDockerString()), docker_version.GetDockerString()]
    occ_filter = _CreateFilterForImageSummaryOccurrences(docker_urls)
    occurrences = ca_requests.ListOccurrences(docker_version.project, occ_filter)
    for occ in occurrences:
        metadata.AddOccurrence(occ, False)
    return metadata