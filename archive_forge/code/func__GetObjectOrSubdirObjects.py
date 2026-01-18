from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os.path
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _GetObjectOrSubdirObjects(object_ref, object_is_subdir=False, client=None):
    """Gets object_ref or the objects under object_ref is it's a subdir."""
    client = client or storage_api.StorageClient()
    objects = []
    target_is_subdir = False
    if not object_is_subdir:
        try:
            client.GetObject(object_ref)
            objects.append(object_ref)
        except apitools_exceptions.HttpNotFoundError:
            target_is_subdir = True
    if target_is_subdir or object_is_subdir:
        target_path = posixpath.join(object_ref.name, '')
        subdir_objects = client.ListBucket(object_ref.bucket_ref, target_path)
        for obj in subdir_objects:
            if object_is_subdir and obj.name == object_ref.name:
                continue
            objects.append(storage_util.ObjectReference.FromName(object_ref.bucket, obj.name))
    return objects