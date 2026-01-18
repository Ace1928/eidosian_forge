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
class GCSPathExpander(PathExpander):
    """Implements path expansion for gs:// formatted resource strings."""

    def __init__(self):
        super(GCSPathExpander, self).__init__('/')
        self._client = storage_api.StorageClient()
        self._objects = {}
        self._object_details = {}

    def GetSortedObjectDetails(self, object_paths):
        """Gets all the details for the given paths and returns them sorted.

    Args:
      object_paths: [str], A list of gs:// object or directory paths.

    Returns:
      [{path, data}], A list of dicts with the keys path and data. Path is the
      gs:// path to the object or directory. Object paths will not end in a '/'
      and directory paths will. The data is either a storage.Object message (for
      objects) or a storage_util.ObjectReference for directories. The sort
      order is alphabetical with all directories first and then all objects.
    """
        all_data = []
        for path in object_paths:
            is_obj, data = self._GetObjectDetails(path)
            path = path if is_obj else path + '/'
            all_data.append((is_obj, {'path': path, 'data': data}))
        all_data = sorted(all_data, key=lambda o: (o[0], o[1]['path']))
        return [d[1] for d in all_data]

    def _GetObjectDetails(self, object_path):
        """Gets the actual object data for a given GCS path.

    Args:
      object_path: str, The gs:// path to an object or directory.

    Returns:
      (bool, data), Where element 0 is True if the path is an object, False if
      a directory and where data is either a storage.Object message (for
      objects) or a storage_util.ObjectReference for directories.
    """
        details = self._object_details.get(object_path)
        if details:
            return (True, details)
        else:
            return (False, storage_util.ObjectReference.FromUrl(object_path, allow_empty_object=True))

    def AbsPath(self, path):
        if not path.startswith('gs://'):
            raise ValueError('GCS paths must be absolute (starting with gs://)')
        return path

    def IsFile(self, path):
        exists, is_dir = self._Exists(path)
        return exists and (not is_dir)

    def IsDir(self, path):
        exists, is_dir = self._Exists(path)
        return exists and is_dir

    def Exists(self, path):
        exists, _ = self._Exists(path)
        return exists

    def _Exists(self, path):
        if self._IsRoot(path):
            return (True, True)
        path = path.rstrip('/')
        obj_ref = storage_util.ObjectReference.FromUrl(path, allow_empty_object=True)
        self._LoadObjectsIfMissing(obj_ref.bucket_ref)
        if obj_ref.bucket in self._objects:
            if not obj_ref.name:
                return (True, True)
            if obj_ref.name in self._objects[obj_ref.bucket]:
                return (True, False)
            dir_name = self._GetDirString(obj_ref.name)
            for i in self._objects[obj_ref.bucket]:
                if i.startswith(dir_name):
                    return (True, True)
        return (False, False)

    def ListDir(self, path):
        if self._IsRoot(path):
            for b in self._client.ListBuckets(project=properties.VALUES.core.project.Get(required=True)):
                yield b.name
            return
        obj_ref = storage_util.ObjectReference.FromUrl(path, allow_empty_object=True)
        self._LoadObjectsIfMissing(obj_ref.bucket_ref)
        dir_name = self._GetDirString(obj_ref.name)
        parent_dir_length = len(dir_name)
        seen = set()
        for obj_name in self._objects[obj_ref.bucket]:
            if obj_name.startswith(dir_name):
                suffix = obj_name[parent_dir_length:]
                result = suffix.split(self._sep)[0]
                if result not in seen:
                    seen.add(result)
                    yield result

    def Join(self, path1, path2):
        if self._IsRoot(path1):
            return 'gs://' + path2.lstrip(self._sep)
        return path1.rstrip(self._sep) + self._sep + path2.lstrip(self._sep)

    def _IsRoot(self, path):
        return path == 'gs://' or path == 'gs:'

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

    def _GetDirString(self, path):
        if path and (not path.endswith(self._sep)):
            return path + self._sep
        return path

    def _FormatPath(self, path):
        path = super(GCSPathExpander, self)._FormatPath(path)
        return 'gs://' if path == 'gs:/' else path