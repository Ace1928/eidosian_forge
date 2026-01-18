from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _ValidateAndParseInputTableName(table_name):
    """Validate BigQuery table name format and returned parsed components."""
    name_parts = table_name.split('.')
    if len(name_parts) != 3:
        raise BigQueryTableNameError('Invalid BigQuery table name [{}]. BigQuery tables are uniquely identified by their project_id, dataset_id, and table_id in the format `<project_id>.<dataset_id>.<table_id>`.'.format(table_name))
    return name_parts