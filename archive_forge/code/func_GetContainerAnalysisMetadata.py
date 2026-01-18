from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def GetContainerAnalysisMetadata(docker_version, args):
    """Retrieves metadata for a docker image."""
    metadata = ContainerAnalysisMetadata()
    docker_urls = ['https://{}'.format(docker_version.GetDockerString()), docker_version.GetDockerString()]
    occ_filter = _CreateFilterFromImagesDescribeArgs(docker_urls, args)
    if occ_filter is None:
        return metadata
    occurrences = ca_requests.ListOccurrences(docker_version.project, occ_filter)
    include_build = args.show_build_details or args.show_all_metadata or args.metadata_filter
    for occ in occurrences:
        metadata.AddOccurrence(occ, include_build)
    if metadata.vulnerability.vulnerabilities:
        vuln_summary = ca_requests.GetVulnerabilitySummary(docker_version.project, filter_util.ContainerAnalysisFilter().WithResources(docker_urls).GetFilter())
        metadata.vulnerability.AddSummary(vuln_summary)
    return metadata