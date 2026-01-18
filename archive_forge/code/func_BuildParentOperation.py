from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def BuildParentOperation(project_id, location_id, index_endpoint_id, operation_id):
    """Build multi-parent operation."""
    return ParseIndexEndpointOperation('projects/{}/locations/{}/indexEndpoints/{}/operations/{}'.format(project_id, location_id, index_endpoint_id, operation_id))