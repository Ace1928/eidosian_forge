from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import fnmatch
import os
import re
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _LoadObjectsIfMissing(self, bucket_ref):
    objects = self._objects.get(bucket_ref.bucket)
    if objects is None:
        try:
            objects = self._client.ListBucket(bucket_ref)
            object_names = set()
            for o in objects:
                full_path = 'gs://' + self.Join(bucket_ref.bucket, o.name)
                self._object_details[full_path] = o
                object_names.add(o.name)
            self._objects.setdefault(bucket_ref.bucket, set()).update(object_names)
        except storage_api.BucketNotFoundError:
            pass