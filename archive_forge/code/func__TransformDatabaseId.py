from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from cloudsdk.google.protobuf import descriptor_pb2
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core.util import files
def _TransformDatabaseId(resource):
    """Gets database ID depending on operation type."""
    metadata = resource.get('metadata')
    base_type = 'type.googleapis.com/google.spanner.admin.database.v1.{}'
    op_type = metadata.get('@type')
    if op_type == base_type.format('RestoreDatabaseMetadata') or op_type == base_type.format('OptimizeRestoredDatabaseMetadata'):
        return metadata.get('name')
    return metadata.get('database')