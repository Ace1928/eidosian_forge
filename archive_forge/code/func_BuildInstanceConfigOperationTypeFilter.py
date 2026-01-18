from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def BuildInstanceConfigOperationTypeFilter(op_type):
    """Builds the filter for the different instance config operation metadata types."""
    if op_type is None:
        return ''
    base_string = 'metadata.@type:type.googleapis.com/google.spanner.admin.instance.v1.'
    if op_type == 'INSTANCE_CONFIG_CREATE':
        return base_string + 'CreateInstanceConfigMetadata'
    if op_type == 'INSTANCE_CONFIG_UPDATE':
        return base_string + 'UpdateInstanceConfigMetadata'