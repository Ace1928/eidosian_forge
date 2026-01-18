from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
def GetVulnerabilities(project, resource, query):
    """Given image, return vulnerabilities."""
    filter_kinds = ['VULNERABILITY']
    filter_ca = filter_util.ContainerAnalysisFilter()
    filter_ca.WithKinds(filter_kinds)
    filter_ca.WithResources([resource])
    filter_ca.WithCustomFilter(query)
    occurrences = requests.ListOccurrencesWithFilters(project, filter_ca.GetChunkifiedFilters())
    return occurrences