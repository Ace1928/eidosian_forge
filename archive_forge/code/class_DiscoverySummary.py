from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
class DiscoverySummary:
    """DiscoverySummary holds image vulnerability discovery information."""

    def __init__(self):
        self.discovery = []

    def AddOccurrence(self, occ):
        self.discovery.append(occ)