from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _GcsBucketConfig(_BucketConfig):
    """Holder for GCS-specific bucket fields.

  See superclass for remaining attributes.

  Subclass Attributes:
    autoclass_terminal_storage_class (str|None): The storage class that
      objects in the bucket eventually transition to if they are not '
      read for a certain length of time.
    default_encryption_key (str|None): A key used to encrypt objects
      added to the bucket.
    default_event_based_hold (bool|None): Determines if event-based holds will
      automatically be applied to new objects in bucket.
    default_object_acl_file_path (str|None): File path to default object ACL
      file.
    default_object_acl_grants_to_add (list[dict]|None): Add default object ACL
      grants to an entity for objects in the bucket.
    default_object_acl_grants_to_remove (list[str]|None): Remove default object
      ACL grants.
    default_storage_class (str|None): Storage class assigned to objects in the
      bucket by default.
    enable_autoclass (bool|None): Enable, disable, or don't do anything to the
      autoclass feature. Autoclass automatically changes object storage class
      based on usage.
    enable_per_object_retention (bool|None): Enable the object retention for the
      bucket.
    enable_hierarchical_namespace (bool|None): Enable heirarchical namespace
    during bucket creation.
    placement (list|None): Dual-region of bucket.
    public_access_prevention (bool|None): Blocks public access to bucket.
      See docs for specifics:
      https://cloud.google.com/storage/docs/public-access-prevention
    recovery_point_objective (str|None): Specifies the replication setting for
      dual-region and multi-region buckets.
    retention_period (int|None): Minimum retention period in seconds for objects
      in a bucket. Attempts to delete an object earlier will be denied.
    soft_delete_duration (int|None): Number of seconds objects are preserved and
      restorable after deletion in a bucket with soft delete enabled.
    uniform_bucket_level_access (bool|None):
      Determines if the IAM policies will apply to every object in bucket.
  """

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None, autoclass_terminal_storage_class=None, cors_file_path=None, default_encryption_key=None, default_event_based_hold=None, default_object_acl_file_path=None, default_object_acl_grants_to_add=None, default_object_acl_grants_to_remove=None, default_storage_class=None, enable_autoclass=None, enable_per_object_retention=None, enable_hierarchical_namespace=None, labels_file_path=None, labels_to_append=None, labels_to_remove=None, lifecycle_file_path=None, location=None, log_bucket=None, log_object_prefix=None, placement=None, public_access_prevention=None, recovery_point_objective=None, requester_pays=None, retention_period=None, retention_period_to_be_locked=None, soft_delete_duration=None, uniform_bucket_level_access=None, versioning=None, web_error_page=None, web_main_page_suffix=None):
        super(_GcsBucketConfig, self).__init__(acl_file_path, acl_grants_to_add, acl_grants_to_remove, cors_file_path, labels_file_path, labels_to_append, labels_to_remove, lifecycle_file_path, location, log_bucket, log_object_prefix, requester_pays, versioning, web_error_page, web_main_page_suffix)
        self.autoclass_terminal_storage_class = autoclass_terminal_storage_class
        self.default_encryption_key = default_encryption_key
        self.default_event_based_hold = default_event_based_hold
        self.default_object_acl_file_path = default_object_acl_file_path
        self.default_object_acl_grants_to_add = default_object_acl_grants_to_add
        self.default_object_acl_grants_to_remove = default_object_acl_grants_to_remove
        self.default_storage_class = default_storage_class
        self.enable_autoclass = enable_autoclass
        self.enable_per_object_retention = enable_per_object_retention
        self.enable_hierarchical_namespace = enable_hierarchical_namespace
        self.placement = placement
        self.public_access_prevention = public_access_prevention
        self.recovery_point_objective = recovery_point_objective
        self.requester_pays = requester_pays
        self.retention_period = retention_period
        self.retention_period_to_be_locked = retention_period_to_be_locked
        self.soft_delete_duration = soft_delete_duration
        self.uniform_bucket_level_access = uniform_bucket_level_access

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return super(_GcsBucketConfig, self).__eq__(other) and self.autoclass_terminal_storage_class == other.autoclass_terminal_storage_class and (self.default_encryption_key == other.default_encryption_key) and (self.default_event_based_hold == other.default_event_based_hold) and (self.default_object_acl_grants_to_add == other.default_object_acl_grants_to_add) and (self.default_object_acl_grants_to_remove == other.default_object_acl_grants_to_remove) and (self.default_storage_class == other.default_storage_class) and (self.enable_autoclass == other.enable_autoclass) and (self.enable_per_object_retention == other.enable_per_object_retention) and (self.enable_hierarchical_namespace == other.enable_hierarchical_namespace) and (self.placement == other.placement) and (self.public_access_prevention == other.public_access_prevention) and (self.recovery_point_objective == other.recovery_point_objective) and (self.requester_pays == other.requester_pays) and (self.retention_period == other.retention_period) and (self.retention_period_to_be_locked == other.retention_period_to_be_locked) and (self.soft_delete_duration == other.soft_delete_duration) and (self.uniform_bucket_level_access == other.uniform_bucket_level_access)