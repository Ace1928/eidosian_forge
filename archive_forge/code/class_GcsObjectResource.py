from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
class GcsObjectResource(resource_reference.ObjectResource):
    """API-specific subclass for handling metadata.

  Additional GCS Attributes:
    storage_class_update_time (datetime|None): Storage class update time.
    hard_delete_time (datetime|None): Time that soft-deleted objects will be
      permanently deleted.
    retention_settings (dict|None): Contains retention settings for individual
      object.
    soft_delete_time (datetime|None): Time that object was soft-deleted.
  """

    def __init__(self, storage_url_object, acl=None, cache_control=None, component_count=None, content_disposition=None, content_encoding=None, content_language=None, content_type=None, crc32c_hash=None, creation_time=None, custom_fields=None, custom_time=None, decryption_key_hash_sha256=None, encryption_algorithm=None, etag=None, event_based_hold=None, hard_delete_time=None, kms_key=None, md5_hash=None, metadata=None, metageneration=None, noncurrent_time=None, retention_expiration=None, retention_settings=None, size=None, soft_delete_time=None, storage_class=None, storage_class_update_time=None, temporary_hold=None, update_time=None):
        """Initializes GcsObjectResource."""
        super(GcsObjectResource, self).__init__(storage_url_object, acl, cache_control, component_count, content_disposition, content_encoding, content_language, content_type, crc32c_hash, creation_time, custom_fields, custom_time, decryption_key_hash_sha256, encryption_algorithm, etag, event_based_hold, kms_key, md5_hash, metadata, metageneration, noncurrent_time, retention_expiration, size, storage_class, temporary_hold, update_time)
        self.hard_delete_time = hard_delete_time
        self.retention_settings = retention_settings
        self.soft_delete_time = soft_delete_time
        self.storage_class_update_time = storage_class_update_time

    def __eq__(self, other):
        return super(GcsObjectResource, self).__eq__(other) and self.hard_delete_time == other.hard_delete_time and (self.retention_settings == other.retention_settings) and (self.soft_delete_time == other.soft_delete_time) and (self.storage_class_update_time == other.storage_class_update_time)

    def get_json_dump(self):
        return _get_json_dump(self)

    def is_encrypted(self):
        cmek_in_metadata = self.metadata.kmsKeyName if self.metadata else False
        return cmek_in_metadata or self.decryption_key_hash_sha256

    def get_formatted_acl(self):
        """See base class."""
        return {full_resource_formatter.ACL_KEY: _get_formatted_acl(self.acl)}