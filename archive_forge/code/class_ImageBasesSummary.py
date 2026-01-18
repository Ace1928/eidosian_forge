from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
class ImageBasesSummary(SummaryResolver):
    """PackageVulnerabilitiesSummary has information about image basis."""

    def __init__(self):
        self.base_images = []

    def add_record(self, occ):
        self.base_images.append(occ)