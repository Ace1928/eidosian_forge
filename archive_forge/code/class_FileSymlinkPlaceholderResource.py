from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class FileSymlinkPlaceholderResource(FileObjectResource):
    """A file to a symlink that should be preserved as a placeholder.

  Attributes:
    Refer to super class.
  """

    def __init__(self, storage_url_object, md5_hash=None):
        """Initializes resource. Args are a subset of attributes."""
        super(FileSymlinkPlaceholderResource, self).__init__(storage_url_object, md5_hash, True)

    @property
    def size(self):
        """Returns the length of the symlink target to be used as a placeholder."""
        return len(os.readlink(self.storage_url.object_name).encode('utf-8'))

    @property
    def is_symlink(self):
        return True