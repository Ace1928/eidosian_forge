from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def GetServiceNameFromArg(service):
    if not service:
        return None
    service_ref = resources.REGISTRY.Parse(service, collection='servicemanagement.services')
    return service_ref.serviceName