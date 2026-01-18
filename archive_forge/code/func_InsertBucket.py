from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def InsertBucket(self, project_id, bucket):
    """Inserts the bucket object as a GCS bucket associated with the project.

    Args:
      project_id: The project string to save the bucket to.
      bucket: The bucket message object to insert.

    Raises:
      HttpError: with status_code 409 if the bucket already exists.
    """
    messages = self._storage_client.MESSAGES_MODULE
    self._storage_client.buckets.Insert(messages.StorageBucketsInsertRequest(bucket=bucket, project=project_id))