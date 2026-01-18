from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _CreateFilterForMaven(maven_resource):
    """Builds filters for containeranalysis APIs for Maven Artifacts."""
    occ_filter = filter_util.ContainerAnalysisFilter()
    filter_kinds = ['VULNERABILITY', 'DISCOVERY']
    occ_filter.WithKinds(filter_kinds)
    occ_filter.WithResources([maven_resource])
    return occ_filter.GetFilter()