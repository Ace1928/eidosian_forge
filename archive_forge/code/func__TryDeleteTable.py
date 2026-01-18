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
def _TryDeleteTable(dataset_id, table_id, project_id):
    """Try to delete a dataset, propagating error on failure."""
    client = GetApiClient()
    service = client.tables
    delete_request_type = GetApiMessage('BigqueryTablesDeleteRequest')
    delete_request = delete_request_type(datasetId=dataset_id, tableId=table_id, projectId=project_id)
    service.Delete(delete_request)
    log.info('Deleted table [{}:{}:{}]'.format(project_id, dataset_id, table_id))