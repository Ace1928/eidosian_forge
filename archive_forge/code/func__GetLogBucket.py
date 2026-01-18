from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import json
import time
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.diagnose import diagnose_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _GetLogBucket(self, project_id):
    """Locates or creates the GCS Bucket for logs associated with the project.

    Args:
      project_id: The id number of the project the bucket is associated with.
    Returns:
      The name of the GCS Bucket.
    """
    project_number = self._GetProjectNumber(project_id)
    bucket_name = '{}_{}'.format(_GCS_LOGS_BUCKET_PREFIX, project_number)
    bucket = self._diagnose_client.FindBucket(project_id, bucket_name)
    if bucket is None:
        bucket = self._diagnose_client.CreateBucketWithLifecycle(days_to_live=10)
        bucket.name = bucket_name
        suffix = 0
        bucket_name_taken = True
        while bucket_name_taken:
            try:
                self._diagnose_client.InsertBucket(project_id, bucket)
                bucket_name_taken = False
            except HttpError as e:
                if e.status_code != 409:
                    raise e
                bucket.name = '{}_{}'.format(bucket_name, suffix)
                suffix += 1
    return bucket.name