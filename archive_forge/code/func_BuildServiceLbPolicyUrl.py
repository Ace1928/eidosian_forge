from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def BuildServiceLbPolicyUrl(project_name, location, policy_name, release_track):
    return BuildFullResourceUrlForProjectBasedResource(base_uri=network_services.GetApiBaseUrl(release_track), project_name=project_name, location=location, collection_name='serviceLbPolicies', resource_name=policy_name)