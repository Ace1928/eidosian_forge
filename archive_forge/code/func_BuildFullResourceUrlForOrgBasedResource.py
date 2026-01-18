from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def BuildFullResourceUrlForOrgBasedResource(base_uri, org_id, collection_name, resource_name):
    """Note: base_uri ends with slash."""
    return BuildFullResourceUrl(base_uri, 'organizations', org_id, 'global', collection_name, resource_name)