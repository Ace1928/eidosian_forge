from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
def GetLatestScan(project, resource):
    """Given project and resource, get the last time it was scanned."""
    filter_kinds = ['DISCOVERY']
    filter_ca = filter_util.ContainerAnalysisFilter()
    filter_ca.WithKinds(filter_kinds)
    filter_ca.WithResources([resource])
    occurrences = requests.ListOccurrencesWithFilters(project, filter_ca.GetChunkifiedFilters())
    latest_scan = None
    for occ in occurrences:
        if latest_scan is None:
            latest_scan = occ
            continue
        try:
            if latest_scan.discovery.lastScanTime < occ.discovery.lastScanTime:
                latest_scan = occ
        except AttributeError:
            continue
    return latest_scan