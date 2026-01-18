from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _GcsObjectConfig(_ObjectConfig):
    """Arguments object for requests with custom GCS parameters.

  See superclass for additional attributes.

  Attributes:
    event_based_hold (bool|None): An event-based hold should be placed on an
      object.
    custom_time (datetime|None): Custom time user can set.
    retain_until (datetime|None): Time to retain the object until.
    retention_mode (flags.RetentionMode|None|CLEAR): The key that should
      be used to set the retention mode policy in GCS or clear retention (the
      string user_request_args_factory.CLEAR).
    temporary_hold (bool|None): A temporary hold should be placed on an object.
  """

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None, cache_control=None, content_disposition=None, content_encoding=None, content_language=None, content_type=None, custom_fields_to_set=None, custom_fields_to_remove=None, custom_fields_to_update=None, custom_time=None, decryption_key=None, encryption_key=None, event_based_hold=None, md5_hash=None, retain_until=None, retention_mode=None, size=None, temporary_hold=None):
        super(_GcsObjectConfig, self).__init__(acl_file_path=acl_file_path, acl_grants_to_add=acl_grants_to_add, acl_grants_to_remove=acl_grants_to_remove, cache_control=cache_control, content_disposition=content_disposition, content_encoding=content_encoding, content_language=content_language, content_type=content_type, custom_fields_to_set=custom_fields_to_set, custom_fields_to_remove=custom_fields_to_remove, custom_fields_to_update=custom_fields_to_update, decryption_key=decryption_key, encryption_key=encryption_key, md5_hash=md5_hash, size=size)
        self.custom_time = custom_time
        self.event_based_hold = event_based_hold
        self.retain_until = retain_until
        self.retention_mode = retention_mode
        self.temporary_hold = temporary_hold

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_GcsObjectConfig, self).__eq__(other) and self.custom_time == other.custom_time and (self.event_based_hold == other.event_based_hold) and (self.retain_until == other.retain_until) and (self.retention_mode == other.retention_mode) and (self.temporary_hold == other.temporary_hold)