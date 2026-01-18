from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _CreateFilterForImageSummaryOccurrences(images):
    """Builds filters for containeranalysis APIs for build and SBOM occurrences."""
    occ_filter = filter_util.ContainerAnalysisFilter()
    filter_kinds = ['BUILD', 'SBOM_REFERENCE']
    occ_filter.WithKinds(filter_kinds)
    occ_filter.WithResources(images)
    return occ_filter.GetFilter()