from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def GetContainerAnalysisMetadataForImages(repo_or_image, occurrence_filter, images):
    """Retrieves metadata for all images with a given path prefix.

  The prefix may initially be used to resolve to a list of images if
  --show-occurrences-from is used.
  To account for cases where there is or isn't a list of images,
  this always filters on both prefix and the list of images. In both of
  those cases, the lookup is for both the case where there is and isn't
  an https prefix, in both the prefixes and in the images list.

  Args:
    repo_or_image: The repository originally given by the user.
    occurrence_filter: The repository originally given by the user.
    images: The list of images that matched the prefix, without https prepended.

  Returns:
    The metadata about the given images.
  """
    metadata = collections.defaultdict(ContainerAnalysisMetadata)
    prefixes = ['https://{}'.format(repo_or_image.GetDockerString()), repo_or_image.GetDockerString()]
    image_urls = images + ['https://{}'.format(img) for img in images]
    occ_filters = _CreateFilterForImages(prefixes, occurrence_filter, image_urls)
    occurrences = ca_requests.ListOccurrencesWithFilters(repo_or_image.project, occ_filters)
    for occ in occurrences:
        metadata.setdefault(occ.resourceUri, ContainerAnalysisMetadata()).AddOccurrence(occ)
    summary_filters = filter_util.ContainerAnalysisFilter().WithResourcePrefixes(prefixes).WithResources(image_urls).GetChunkifiedFilters()
    summaries = ca_requests.GetVulnerabilitySummaryWithFilters(repo_or_image.project, summary_filters)
    for summary in summaries:
        for count in summary.counts:
            metadata.setdefault(count.resourceUri, ContainerAnalysisMetadata()).vulnerability.AddCount(count)
    return metadata