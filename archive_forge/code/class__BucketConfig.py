from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _BucketConfig(_ResourceConfig):
    """Holder for generic bucket fields.

  More attributes may exist on parent class.

  Attributes:
    cors_file_path (None|str): Path to file with CORS settings.
    labels_file_path (None|str): Path to file with labels settings.
    labels_to_append (None|Dict): Labels to add to a bucket.
    labels_to_remove (None|List[str]): Labels to remove from a bucket.
    lifecycle_file_path (None|str): Path to file with lifecycle settings.
    location (str|None): Location of bucket.
    log_bucket (str|None): Destination bucket for current bucket's logs.
    log_object_prefix (str|None): Prefix for objects containing logs.
    requester_pays (bool|None): If set requester pays all costs related to
      accessing the bucket and its objects.
    versioning (None|bool): Whether to turn on object versioning in a bucket.
    web_error_page (None|str): Error page address if bucket is being used
      to host a website.
    web_main_page_suffix (None|str): Suffix of main page address if bucket is
      being used to host a website.
  """

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None, cors_file_path=None, labels_file_path=None, labels_to_append=None, labels_to_remove=None, lifecycle_file_path=None, location=None, log_bucket=None, log_object_prefix=None, requester_pays=None, versioning=None, web_error_page=None, web_main_page_suffix=None):
        super(_BucketConfig, self).__init__(acl_file_path, acl_grants_to_add, acl_grants_to_remove)
        self.location = location
        self.cors_file_path = cors_file_path
        self.labels_file_path = labels_file_path
        self.labels_to_append = labels_to_append
        self.labels_to_remove = labels_to_remove
        self.lifecycle_file_path = lifecycle_file_path
        self.log_bucket = log_bucket
        self.log_object_prefix = log_object_prefix
        self.requester_pays = requester_pays
        self.versioning = versioning
        self.web_error_page = web_error_page
        self.web_main_page_suffix = web_main_page_suffix

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_BucketConfig, self).__eq__(other) and self.cors_file_path == other.cors_file_path and (self.labels_file_path == other.labels_file_path) and (self.labels_to_append == other.labels_to_append) and (self.labels_to_remove == other.labels_to_remove) and (self.lifecycle_file_path == other.lifecycle_file_path) and (self.location == other.location) and (self.log_bucket == other.log_bucket) and (self.log_object_prefix == other.log_object_prefix) and (self.requester_pays == other.requester_pays) and (self.versioning == other.versioning) and (self.web_error_page == other.web_error_page) and (self.web_main_page_suffix == other.web_main_page_suffix)