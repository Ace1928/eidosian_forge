from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def EnsureBucketExists(self, bucket_name):
    """Create a GCS bucket if it doesn't already exist.

    Args:
      bucket_name: the name of the GCS bucket to create if it doesn't exist.

    Raises:
      BadFileException if the bucket name is malformed, the user does not
        have access rights to the bucket, or the bucket can't be created.
    """
    get_req = self._storage_messages.StorageBucketsGetRequest(bucket=bucket_name)
    try:
        self._storage_client.buckets.Get(get_req)
        return
    except apitools_exceptions.HttpError as err:
        code, err_msg = util.GetErrorCodeAndMessage(err)
        if code != HTTP_NOT_FOUND:
            raise exceptions.BadFileException('Could not access bucket [{b}]. Response error {c}: {e}. Please supply a valid bucket name or use the default bucket provided by Firebase Test Lab.'.format(b=bucket_name, c=code, e=err_msg))
    log.status.Print('Creating results bucket [{g}{b}] in project [{p}].'.format(g=GCS_PREFIX, b=bucket_name, p=self._project))
    bucket_req = self._storage_messages.StorageBucketsInsertRequest
    acl = bucket_req.PredefinedAclValueValuesEnum.projectPrivate
    objacl = bucket_req.PredefinedDefaultObjectAclValueValuesEnum.projectPrivate
    insert_req = self._storage_messages.StorageBucketsInsertRequest(bucket=self._storage_messages.Bucket(name=bucket_name), predefinedAcl=acl, predefinedDefaultObjectAcl=objacl, project=self._project)
    try:
        self._storage_client.buckets.Insert(insert_req)
        return
    except apitools_exceptions.HttpError as err:
        code, err_msg = util.GetErrorCodeAndMessage(err)
        if code == HTTP_FORBIDDEN:
            msg = 'Permission denied while creating bucket [{b}]. Is billing enabled for project: [{p}]?'.format(b=bucket_name, p=self._project)
        else:
            msg = 'Failed to create bucket [{b}] {e}'.format(b=bucket_name, e=util.GetError(err))
        raise exceptions.BadFileException(msg)