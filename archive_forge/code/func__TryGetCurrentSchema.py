from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def _TryGetCurrentSchema(dataset_id, table_id, project_id):
    """Try to retrieve the current BigQuery TableSchema for a table_ref.

    Tries to fetch the schema of an existing table. Raises SchemaUpdateError if
    table is not found or if table is not of type 'TABLE'.

  Args:
    dataset_id: the dataset id containing the table.
    table_id: the table id for the table.
    project_id: the project id containing the dataset and table.


  Returns:
    schema: the table schema object

  Raises:
    SchemaUpdateError: table not found or invalid table type.
  """
    client = GetApiClient()
    service = client.tables
    get_request_type = GetApiMessage('BigqueryTablesGetRequest')
    get_request = get_request_type(datasetId=dataset_id, tableId=table_id, projectId=project_id)
    try:
        table = service.Get(get_request)
        if not table or table.type != 'TABLE':
            raise SchemaUpdateError('Schema modifications only supported on TABLE objects received [{}]'.format(table))
    except apitools_exceptions.HttpNotFoundError:
        raise SchemaUpdateError('Table with id [{}:{}:{}] not found.'.format(project_id, dataset_id, table_id))
    return table.schema