from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import iam_command_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import set_iam_policy_task
def _set_iam_policy_task_iterator(url_strings, policy):
    """Generates SetIamPolicyTask's for execution."""
    for url_string in url_strings:
        for resource in wildcard_iterator.get_wildcard_iterator(url_string, fields_scope=cloud_api.FieldsScope.SHORT):
            yield set_iam_policy_task.SetBucketIamPolicyTask(resource.storage_url, policy)