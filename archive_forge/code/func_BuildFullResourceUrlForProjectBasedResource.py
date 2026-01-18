from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def BuildFullResourceUrlForProjectBasedResource(base_uri, project_name, location, collection_name, resource_name):
    """Note: base_uri ends with slash."""
    return BuildFullResourceUrl(base_uri, 'projects', project_name, location, collection_name, resource_name)