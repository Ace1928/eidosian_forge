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
def ProcessSchemaUpdate(ref, args, request):
    """Process schema Updates (additions/mode changes) for the request.

  Retrieves the current table schema for ref and attempts to merge in the schema
  provided in the requests. This is necessary since the API backend does not
  handle PATCH semantics for schema updates (e.g. process the deltas) so we must
  always send the fully updated schema in the requests.

  Args:
    ref: resource reference for table.
    args: argparse namespace for requests
    request: BigqueryTablesPatchRequest object


  Returns:
    request: updated requests

  Raises:
    SchemaUpdateError: table not found or invalid an schema change.
  """
    table = request.table
    relaxed_columns = args.relax_columns
    if not table.schema and (not relaxed_columns):
        return request
    original_schema = _TryGetCurrentSchema(ref.Parent().Name(), ref.Name(), ref.projectId)
    new_schema_columns = table.schema
    updated_fields = _GetUpdatedSchema(original_schema, new_schema_columns, relaxed_columns)
    table_schema_type = GetApiMessage('TableSchema')
    request.table.schema = table_schema_type(fields=updated_fields)
    return request