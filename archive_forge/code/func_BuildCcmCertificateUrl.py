from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def BuildCcmCertificateUrl(project_name, location, certificate_name):
    base_uri = resources.GetApiBaseUrl('certificatemanager', 'v1') or CERTIFICATE_MANAGER_BASE_API
    return BuildFullResourceUrlForProjectBasedResource(base_uri=base_uri, project_name=project_name, location=location, collection_name='certificates', resource_name=certificate_name)