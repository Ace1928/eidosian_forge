from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _GcsRequestConfig(_RequestConfig):
    """Holder for GCS-specific API request parameters.

  See superclass for additional attributes.

  Attributes:
    gzip_settings (user_request_args_factory.GzipSettings): Contains settings
      for gzipping uploaded files.
    no_clobber (bool): Do not copy if destination resource already exists.
    override_unlocked_retention (bool|None): Needed as confirmation for some
      changes to object retention policies.
    precondition_generation_match (int|None): Perform request only if generation
      of target object matches the given integer. Ignored for bucket requests.
    precondition_metageneration_match (int|None): Perform request only if
      metageneration of target object/bucket matches the given integer.
  """

    def __init__(self, gzip_settings=None, no_clobber=None, override_unlocked_retention=None, precondition_generation_match=None, precondition_metageneration_match=None, predefined_acl_string=None, predefined_default_object_acl_string=None, resource_args=None):
        super(_GcsRequestConfig, self).__init__(predefined_acl_string=predefined_acl_string, predefined_default_object_acl_string=predefined_default_object_acl_string, resource_args=resource_args)
        self.gzip_settings = gzip_settings
        self.no_clobber = no_clobber
        self.override_unlocked_retention = override_unlocked_retention
        self.precondition_generation_match = precondition_generation_match
        self.precondition_metageneration_match = precondition_metageneration_match

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_GcsRequestConfig, self).__eq__(other) and self.gzip_settings == other.gzip_settings and (self.no_clobber == other.no_clobber) and (self.override_unlocked_retention == other.override_unlocked_retention) and (self.precondition_generation_match == other.precondition_generation_match) and (self.precondition_metageneration_match == other.precondition_metageneration_match)