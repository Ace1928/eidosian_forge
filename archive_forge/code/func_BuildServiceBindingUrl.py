from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def BuildServiceBindingUrl(project_name, location, binding_name):
    return BuildFullResourceUrlForProjectBasedResource(base_uri=network_services.GetApiBaseUrl(network_services.base.ReleaseTrack.GA), project_name=project_name, location=location, collection_name='serviceBindings', resource_name=binding_name)