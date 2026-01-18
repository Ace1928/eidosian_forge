from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projection_spec
def _GetListPage(self, compute_interconnect_attachments, request):
    response = compute_interconnect_attachments.AggregatedList(request)
    interconnect_attachments_lists = []
    for attachment_in_scope in response.items.additionalProperties:
        interconnect_attachments_lists += attachment_in_scope.value.interconnectAttachments
    return (interconnect_attachments_lists, response.nextPageToken)