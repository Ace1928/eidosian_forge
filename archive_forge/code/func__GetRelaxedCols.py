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
def _GetRelaxedCols(relaxed_columns, orig_schema_map):
    """Change mode to `NULLABLE` for columns in existing schema.

    Tries set mode on existing columns in orig_schema_map to `NULLABLE`. Raises
    SchemaUpdateError if column is not found in orig_schema_map.

  Args:
    relaxed_columns: [string] the list columns to relax required mode for.
    orig_schema_map: {string: TableSchemaField} map of field name to
      TableSchemaField objects representing the original schema.

  Returns:
    updated_schema_map: {string: TableSchemaField} map of field name to
      TableSchemaField objects representing the updated schema.

  Raises:
    SchemaUpdateError: if any of the fields to be relaxed are not in the
      original schema.
  """
    updated_schema_map = orig_schema_map.copy()
    for col in relaxed_columns:
        if col in orig_schema_map:
            updated_schema_map[col].mode = 'NULLABLE'
        else:
            raise SchemaUpdateError(_INVALID_SCHEMA_UPDATE_MESSAGE)
    return updated_schema_map