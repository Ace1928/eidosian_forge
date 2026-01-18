from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.service_directory import base as sd_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
class ServicesClientBeta(ServicesClient):
    """Client for service in the Service Directory API."""

    def __init__(self):
        super(ServicesClientBeta, self).__init__(base.ReleaseTrack.BETA)

    def Create(self, service_ref, metadata=None):
        """Services create request."""
        service = self.msgs.Service(metadata=metadata)
        create_req = self.msgs.ServicedirectoryProjectsLocationsNamespacesServicesCreateRequest(parent=service_ref.Parent().RelativeName(), service=service, serviceId=service_ref.servicesId)
        return self.service.Create(create_req)

    def Update(self, service_ref, metadata=None):
        """Services update request."""
        mask_parts = []
        if metadata:
            mask_parts.append('metadata')
        service = self.msgs.Service(metadata=metadata)
        update_req = self.msgs.ServicedirectoryProjectsLocationsNamespacesServicesPatchRequest(name=service_ref.RelativeName(), service=service, updateMask=','.join(mask_parts))
        return self.service.Patch(update_req)