from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
def TransformResultStatus(resource, undefined=''):
    """Returns the formatted result status.

  Args:
    resource: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The formatted result status.
  """
    messages = core_apis.GetMessagesModule('cloudbuild', 'v2')
    result = apitools_encoding.DictToMessage(resource, messages.Result)
    record_summary = result.recordSummaries[0]
    record_data = hub_client.HubClient.ToPyDict(record_summary.recordData)
    if record_summary.status is not None:
        return record_summary.status
    if 'pipeline_run_status' in record_data or 'task_run_status' in record_data:
        return 'CANCELLED'
    if 'conditions[0].status' in record_data:
        condition = record_data.get('conditions[0].status')
        if condition == 'TRUE':
            return 'SUCCESS'
        if condition == 'FALSE':
            return 'FAILURE'
        if condition == 'UNKNOWN':
            return 'WORKING'
    if 'start_time' in record_data and 'finish_time' not in record_data and ('completion_time' not in record_data):
        return 'WORKING'
    return undefined