from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def CreateBucketIfNotExists(self, bucket, project=None, location=None, check_ownership=True, enable_uniform_level_access=None, cors=None):
    """Create a bucket if it does not already exist.

    If it already exists and is accessible by the current user, this method
    returns.

    Args:
      bucket: str, The storage bucket to be created.
      project: str, The project to use for the API request. If None, current
        Cloud SDK project is used.
      location: str, The bucket location/region.
      check_ownership: bool, Whether to check that the resulting bucket belongs
        to the given project. DO NOT SET THIS TO FALSE if the bucket name can be
        guessed and claimed ahead of time by another user as it enables a name
        squatting exploit.
      enable_uniform_level_access: bool, to enable uniform bucket level access.
        If None, the iamConfiguration object will not be created in the bucket
        creation request, which means that it will use the default values.
      cors: list, A list of CorsValueListEntry objects. The bucket's
        Cross-Origin Resource Sharing (CORS) configuration. If None, no CORS
        configuration will be set.

    Raises:
      api_exceptions.HttpError: If the bucket is not able to be created or is
        not accessible due to permissions.
      BucketInWrongProjectError: If the bucket already exists in a different
        project. This could belong to a malicious user squatting on the bucket
        name.
    """
    project = project or properties.VALUES.core.project.Get(required=True)
    try:
        self.client.buckets.Get(self.messages.StorageBucketsGetRequest(bucket=bucket))
    except api_exceptions.HttpNotFoundError:
        storage_buckets_insert_request = self.messages.StorageBucketsInsertRequest(project=project, bucket=self.messages.Bucket(name=bucket, location=location))
        if enable_uniform_level_access is not None:
            storage_buckets_insert_request.bucket.iamConfiguration = self.messages.Bucket.IamConfigurationValue(uniformBucketLevelAccess=self.messages.Bucket.IamConfigurationValue.UniformBucketLevelAccessValue(enabled=enable_uniform_level_access))
        if cors is not None:
            storage_buckets_insert_request.bucket.cors = cors
        try:
            self.client.buckets.Insert(storage_buckets_insert_request)
        except api_exceptions.HttpConflictError:
            self.client.buckets.Get(self.messages.StorageBucketsGetRequest(bucket=bucket))
        else:
            return
    if not check_ownership:
        return
    bucket_list_req = self.messages.StorageBucketsListRequest(project=project, prefix=bucket)
    bucket_list = self.client.buckets.List(bucket_list_req)
    if not any((b.id == bucket for b in bucket_list.items)):
        raise BucketInWrongProjectError('Unable to create bucket [{}] as it already exists in another project.'.format(bucket))