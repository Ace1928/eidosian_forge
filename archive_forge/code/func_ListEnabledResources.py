from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def ListEnabledResources(self, project):
    request = self.messages.ComputeProjectsGetXpnResourcesRequest(project=project)
    return list_pager.YieldFromList(self.client.projects, request, method='GetXpnResources', batch_size_attribute='maxResults', batch_size=500, field='resources')