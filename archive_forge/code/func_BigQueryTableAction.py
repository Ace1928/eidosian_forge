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
def BigQueryTableAction(table_name):
    """Convert BigQuery formatted table name into GooglePrivacyDlpV2Action.

  Creates a BigQuery output action for a job trigger.

  Args:
    table_name: str, BigQuery table name to create action from in the form
      `<project_id>.<dataset_id>.<table_id>` or `<project_id>.<dataset_id>`.

  Returns:
    GooglePrivacyDlpV2Action, output action for job trigger.

  Raises:
    BigQueryTableNameError if table_name is improperly formatted.
  """
    name_parts = _ValidateAndParseOutputTableName(table_name)
    project_id = name_parts[0]
    data_set_id = name_parts[1]
    table_id = ''
    if len(name_parts) == 3:
        table_id = name_parts[2]
    action_msg = _GetMessageClass('GooglePrivacyDlpV2Action')
    save_findings_config = _GetMessageClass('GooglePrivacyDlpV2SaveFindings')
    output_config = _GetMessageClass('GooglePrivacyDlpV2OutputStorageConfig')
    big_query_table = _GetMessageClass('GooglePrivacyDlpV2BigQueryTable')
    table = big_query_table(datasetId=data_set_id, projectId=project_id, tableId=table_id)
    return action_msg(saveFindings=save_findings_config(outputConfig=output_config(table=table)))