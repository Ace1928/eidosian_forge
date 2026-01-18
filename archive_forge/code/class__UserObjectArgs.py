from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
class _UserObjectArgs(_UserResourceArgs):
    """Contains user flag values affecting cloud object settings."""

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None, cache_control=None, content_disposition=None, content_encoding=None, content_language=None, content_type=None, custom_fields_to_set=None, custom_fields_to_remove=None, custom_fields_to_update=None, custom_time=None, event_based_hold=None, md5_hash=None, preserve_acl=None, retain_until=None, retention_mode=None, storage_class=None, temporary_hold=None):
        """Initializes class, binding flag values to it."""
        super(_UserObjectArgs, self).__init__(acl_file_path, acl_grants_to_add, acl_grants_to_remove)
        self.cache_control = cache_control
        self.content_disposition = content_disposition
        self.content_encoding = content_encoding
        self.content_language = content_language
        self.content_type = content_type
        self.custom_fields_to_set = custom_fields_to_set
        self.custom_fields_to_remove = custom_fields_to_remove
        self.custom_fields_to_update = custom_fields_to_update
        self.custom_time = custom_time
        self.event_based_hold = event_based_hold
        self.md5_hash = md5_hash
        self.preserve_acl = preserve_acl
        self.retain_until = retain_until
        self.retention_mode = retention_mode
        self.storage_class = storage_class
        self.temporary_hold = temporary_hold

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_UserObjectArgs, self).__eq__(other) and self.cache_control == other.cache_control and (self.content_disposition == other.content_disposition) and (self.content_encoding == other.content_encoding) and (self.content_language == other.content_language) and (self.content_type == other.content_type) and (self.custom_fields_to_set == other.custom_fields_to_set) and (self.custom_fields_to_remove == other.custom_fields_to_remove) and (self.custom_fields_to_update == other.custom_fields_to_update) and (self.custom_time == other.custom_time) and (self.event_based_hold == other.event_based_hold) and (self.md5_hash == other.md5_hash) and (self.preserve_acl == other.preserve_acl) and (self.retain_until == other.retain_until) and (self.retention_mode == other.retention_mode) and (self.storage_class == other.storage_class) and (self.temporary_hold == other.temporary_hold)