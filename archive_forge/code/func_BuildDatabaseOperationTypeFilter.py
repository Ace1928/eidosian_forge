from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def BuildDatabaseOperationTypeFilter(op_type):
    """Builds the filter for the different database operation metadata types."""
    if op_type == 'DATABASE':
        return ''
    base_string = 'metadata.@type:type.googleapis.com/google.spanner.admin.database.v1.'
    if op_type == 'DATABASE_RESTORE':
        return '({}OptimizeRestoredDatabaseMetadata) OR ({}RestoreDatabaseMetadata)'.format(base_string, base_string)
    if op_type == 'DATABASE_CREATE':
        return base_string + 'CreateDatabaseMetadata'
    if op_type == 'DATABASE_UPDATE_DDL':
        return base_string + 'UpdateDatabaseDdlMetadata'
    if op_type == 'DATABASE_CHANGE_QUORUM':
        return base_string + 'DatabaseChangeQuorumMetadata'