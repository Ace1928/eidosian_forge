from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
class GcsBucketResource(resource_reference.BucketResource):
    """API-specific subclass for handling metadata.

  Additional GCS Attributes:
    autoclass (dict|None): Autoclass settings for the bucket
    autoclass_enabled_time (datetime|None): Datetime Autoclass feature was
      enabled on bucket. None means the feature is disabled.
    custom_placement_config (dict|None): Dual Region of a bucket.
    default_acl (dict|None): Default object ACLs for the bucket.
    default_kms_key (str|None): Default KMS key for objects in the bucket.
    location_type (str|None): Region, dual-region, etc.
    per_object_retention (dict|None): Contains object retention settings for
      bucket.
    project_number (int|None): The project number to which the bucket belongs
      (different from project name and project ID).
    public_access_prevention (str|None): Public access prevention status.
    rpo (str|None): Recovery Point Objective status.
    satisfies_pzs (bool|None): Zone Separation status.
    soft_delete_policy (dict|None): Soft delete settings for bucket.
    uniform_bucket_level_access (bool|None): True if all objects in the bucket
      share ACLs rather than the default, fine-grain ACL control.
  """

    def __init__(self, storage_url_object, acl=None, autoclass=None, autoclass_enabled_time=None, cors_config=None, creation_time=None, custom_placement_config=None, default_acl=None, default_event_based_hold=None, default_kms_key=None, default_storage_class=None, etag=None, labels=None, lifecycle_config=None, location=None, location_type=None, logging_config=None, metadata=None, metageneration=None, per_object_retention=None, project_number=None, public_access_prevention=None, requester_pays=None, retention_policy=None, rpo=None, satisfies_pzs=None, soft_delete_policy=None, uniform_bucket_level_access=None, update_time=None, versioning_enabled=None, website_config=None):
        """Initializes resource. Args are a subset of attributes."""
        super(GcsBucketResource, self).__init__(storage_url_object, acl=acl, cors_config=cors_config, creation_time=creation_time, default_event_based_hold=default_event_based_hold, default_storage_class=default_storage_class, etag=etag, labels=labels, lifecycle_config=lifecycle_config, location=location, logging_config=logging_config, metageneration=metageneration, metadata=metadata, requester_pays=requester_pays, retention_policy=retention_policy, update_time=update_time, versioning_enabled=versioning_enabled, website_config=website_config)
        self.autoclass = autoclass
        self.autoclass_enabled_time = autoclass_enabled_time
        self.custom_placement_config = custom_placement_config
        self.default_acl = default_acl
        self.default_kms_key = default_kms_key
        self.location_type = location_type
        self.per_object_retention = per_object_retention
        self.project_number = project_number
        self.public_access_prevention = public_access_prevention
        self.rpo = rpo
        self.satisfies_pzs = satisfies_pzs
        self.soft_delete_policy = soft_delete_policy
        self.uniform_bucket_level_access = uniform_bucket_level_access

    @property
    def data_locations(self):
        if self.custom_placement_config:
            return self.custom_placement_config.get('dataLocations')
        return None

    @property
    def retention_period(self):
        if self.retention_policy and self.retention_policy.get('retentionPeriod'):
            return int(self.retention_policy['retentionPeriod'])
        return None

    @property
    def retention_policy_is_locked(self):
        return self.retention_policy and self.retention_policy.get('isLocked', False)

    def __eq__(self, other):
        return super(GcsBucketResource, self).__eq__(other) and self.autoclass == other.autoclass and (self.autoclass_enabled_time == other.autoclass_enabled_time) and (self.custom_placement_config == other.custom_placement_config) and (self.default_acl == other.default_acl) and (self.default_kms_key == other.default_kms_key) and (self.location_type == other.location_type) and (self.per_object_retention == other.per_object_retention) and (self.project_number == other.project_number) and (self.public_access_prevention == other.public_access_prevention) and (self.rpo == other.rpo) and (self.satisfies_pzs == other.satisfies_pzs) and (self.soft_delete_policy == other.soft_delete_policy) and (self.uniform_bucket_level_access == other.uniform_bucket_level_access)

    def get_json_dump(self):
        return _get_json_dump(self)

    def get_formatted_acl(self):
        """See base class."""
        return {full_resource_formatter.ACL_KEY: _get_formatted_acl(self.acl), full_resource_formatter.DEFAULT_ACL_KEY: _get_formatted_acl(self.default_acl)}