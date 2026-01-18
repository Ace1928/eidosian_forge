from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _RequestConfig(object):
    """Holder for parameters shared between cloud providers.

  Provider-specific subclasses may add more attributes.

  Attributes:
    predefined_acl_string (str|None): ACL to set on resource.
    predefined_default_object_acl_string (str|None): Default ACL to set on
      resources.
    preserve_posix (bool|None): Whether to apply source POSIX metadata to
      destination.
    preserve_symlinks (bool|None): Whether symlinks should be preserved rather
      than followed.
    resource_args (_BucketConfig|_ObjectConfig|None): Holds settings for a cloud
      resource.
  """

    def __init__(self, predefined_acl_string=None, predefined_default_object_acl_string=None, preserve_posix=None, preserve_symlinks=None, resource_args=None):
        self.predefined_acl_string = predefined_acl_string
        self.predefined_default_object_acl_string = predefined_default_object_acl_string
        self.preserve_posix = preserve_posix
        self.preserve_symlinks = preserve_symlinks
        self.resource_args = resource_args

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.predefined_acl_string == other.predefined_acl_string and self.predefined_default_object_acl_string == other.predefined_default_object_acl_string and (self.preserve_posix == other.preserve_posix) and (self.preserve_symlinks == other.preserve_symlinks) and (self.resource_args == other.resource_args)

    def __repr__(self):
        return debug_output.generic_repr(self)