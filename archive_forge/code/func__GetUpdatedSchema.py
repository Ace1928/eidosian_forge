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
def _GetUpdatedSchema(original_schema, new_columns=None, relaxed_columns=None):
    """Update original_schema by adding and/or relaxing mode on columns."""
    orig_field_map = {f.name: f for f in original_schema.fields} if original_schema else {}
    if relaxed_columns:
        orig_field_map = _GetRelaxedCols(relaxed_columns, orig_field_map)
    if new_columns:
        orig_field_map = _AddNewColsToSchema(new_columns.fields, orig_field_map)
    return sorted(orig_field_map.values(), key=lambda x: x.name)