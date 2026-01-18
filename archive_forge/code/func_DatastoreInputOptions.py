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
def DatastoreInputOptions(table_name):
    """Convert Datastore arg value into GooglePrivacyDlpV2DatastoreOptions.

  Creates Datastore input options for a job trigger from datastore table name.

  Args:
    table_name: str, Datastore table name to create options from in the form
    `namespace:example-kind` or simply `example-kind`.

  Returns:
    GooglePrivacyDlpV2Action, output action for job trigger.
  """
    data_store_options = _GetMessageClass('GooglePrivacyDlpV2DatastoreOptions')
    kind = _GetMessageClass('GooglePrivacyDlpV2KindExpression')
    partition_id = _GetMessageClass('GooglePrivacyDlpV2PartitionId')
    project = properties.VALUES.core.project.Get(required=True)
    split_name = table_name.split(':')
    if len(split_name) == 2:
        namespace, table = split_name
        kind_exp = kind(name=table)
        partition = partition_id(namespaceId=namespace, projectId=project)
    else:
        kind_exp = kind(name=table_name)
        partition = partition_id(projectId=project)
    return data_store_options(kind=kind_exp, partitionId=partition)