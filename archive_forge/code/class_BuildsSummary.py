from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
class BuildsSummary(SummaryResolver):
    """BuildsSummary has information about builds."""

    def __init__(self):
        self.build_details = []

    def add_record(self, occ):
        self.build_details.append(occ)