from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _ObjectConfig(_ResourceConfig):
    """Holder for storage object settings shared between cloud providers.

  Superclass and provider-specific subclasses may add more attributes.

  Attributes:
    cache_control (str|None): Influences how backend caches requests and
      responses.
    content_disposition (str|None): Information on how content should be
      displayed.
    content_encoding (str|None): How content is encoded (e.g. "gzip").
    content_language (str|None): Content's language (e.g. "en" = "English).
    content_type (str|None): Type of data contained in content (e.g.
      "text/html").
    custom_fields_to_set (dict|None): Custom metadata fields set by user.
    custom_fields_to_remove (dict|None): Custom metadata fields to be removed.
    custom_fields_to_update (dict|None): Custom metadata fields to be added or
      changed.
    decryption_key (encryption_util.EncryptionKey): The key that should be used
      to decrypt information in GCS.
    encryption_key (encryption_util.EncryptionKey|None|CLEAR): The key that
      should be used to encrypt information in GCS or clear encryptions (the
      string user_request_args_factory.CLEAR).
    md5_hash (str|None): MD5 digest to use for validation.
    preserve_acl (bool): Whether or not to preserve existing ACLs on an object
      during a copy or other operation.
    size (int|None): Object size in bytes.
    storage_class (str|None): Storage class for cloud object. If None, will use
      bucket's default.
  """

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None, cache_control=None, content_disposition=None, content_encoding=None, content_language=None, content_type=None, custom_fields_to_set=None, custom_fields_to_remove=None, custom_fields_to_update=None, decryption_key=None, encryption_key=None, md5_hash=None, preserve_acl=None, size=None, storage_class=None):
        super(_ObjectConfig, self).__init__(acl_file_path, acl_grants_to_add, acl_grants_to_remove)
        self.cache_control = cache_control
        self.content_disposition = content_disposition
        self.content_encoding = content_encoding
        self.content_language = content_language
        self.content_type = content_type
        self.custom_fields_to_set = custom_fields_to_set
        self.custom_fields_to_remove = custom_fields_to_remove
        self.custom_fields_to_update = custom_fields_to_update
        self.decryption_key = decryption_key
        self.encryption_key = encryption_key
        self.md5_hash = md5_hash
        self.preserve_acl = preserve_acl
        self.size = size
        self.storage_class = storage_class

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_ObjectConfig, self).__eq__(other) and self.cache_control == other.cache_control and (self.content_disposition == other.content_disposition) and (self.content_encoding == other.content_encoding) and (self.content_language == other.content_language) and (self.content_type == other.content_type) and (self.custom_fields_to_set == other.custom_fields_to_set) and (self.custom_fields_to_remove == other.custom_fields_to_remove) and (self.custom_fields_to_update == other.custom_fields_to_update) and (self.decryption_key == other.decryption_key) and (self.encryption_key == other.encryption_key) and (self.md5_hash == other.md5_hash) and (self.size == other.size) and (self.preserve_acl == other.preserve_acl) and (self.storage_class == other.storage_class)