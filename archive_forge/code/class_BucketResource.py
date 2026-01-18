from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class BucketResource(CloudResource):
    """Class representing a bucket.

  Warning: After being run through through output formatter utils (e.g. in
  `shim_format_util.py`), these fields may all be strings.

  Attributes:
    TYPE_STRING (str): String representing the resource's content type.
    storage_url (StorageUrl): A StorageUrl object representing the bucket.
    name (str): Name of bucket.
    scheme (storage_url.ProviderPrefix): Prefix indicating what cloud provider
      hosts the bucket.
    acl (dict|CloudApiError|None): ACLs dict or predefined-ACL string for the
      bucket. If the API call to fetch the data failed, this can be an error
      string.
    cors_config (dict|CloudApiError|None): CORS configuration for the bucket.
      If the API call to fetch the data failed, this can be an error string.
    creation_time (datetime|None): Bucket's creation time in UTC.
    default_event_based_hold (bool|None): Prevents objects in bucket from being
      deleted. Currently GCS-only but needed for generic copy logic.
    default_storage_class (str|None): Default storage class for objects in
      bucket.
    etag (str|None): HTTP version identifier.
    labels (dict|None): Labels for the bucket.
    lifecycle_config (dict|CloudApiError|None): Lifecycle configuration for
      bucket. If the API call to fetch the data failed, this can be an error
      string.
    location (str|None): Represents region bucket was created in.
      If the API call to fetch the data failed, this can be an error string.
    logging_config (dict|CloudApiError|None): Logging configuration for bucket.
      If the API call to fetch the data failed, this can be an error string.
    metadata (object|dict|None): Cloud-provider specific data type for holding
      bucket metadata.
    metageneration (int|None): The generation of the bucket's metadata.
    requester_pays (bool|CloudApiError|None): "Requester pays" status of bucket.
      If the API call to fetch the data failed, this can be an error string.
    retention_period (int|None): Default time to hold items in bucket before
      before deleting in seconds. Generated from retention_policy.
    retention_policy (dict|None): Info about object retention within bucket.
    retention_policy_is_locked (bool|None): True if a retention policy is
      locked.
    update_time (str|None): Bucket's update time.
    versioning_enabled (bool|CloudApiError|None): Whether past object versions
      are saved. If the API call to fetch the data failed, this can be an error
      string.
    website_config (dict|CloudApiError|None): Website configuration for bucket.
      If the API call to fetch the data failed, this can be an error string.
  """
    TYPE_STRING = 'cloud_bucket'

    def __init__(self, storage_url_object, acl=None, cors_config=None, creation_time=None, default_event_based_hold=None, default_storage_class=None, etag=None, labels=None, lifecycle_config=None, location=None, logging_config=None, metageneration=None, metadata=None, requester_pays=None, retention_policy=None, update_time=None, versioning_enabled=None, website_config=None):
        """Initializes resource. Args are a subset of attributes."""
        super(BucketResource, self).__init__(storage_url_object)
        self.acl = acl
        self.cors_config = cors_config
        self.creation_time = creation_time
        self.default_event_based_hold = default_event_based_hold
        self.default_storage_class = default_storage_class
        self.etag = etag
        self.labels = labels
        self.lifecycle_config = lifecycle_config
        self.location = location
        self.logging_config = logging_config
        self.metadata = metadata
        self.metageneration = metageneration
        self.requester_pays = requester_pays
        self.retention_policy = retention_policy
        self.update_time = update_time
        self.versioning_enabled = versioning_enabled
        self.website_config = website_config

    @property
    def name(self):
        return self.storage_url.bucket_name

    @property
    def retention_period(self):
        return None

    @property
    def retention_policy_is_locked(self):
        return None

    def __eq__(self, other):
        return super(BucketResource, self).__eq__(other) and self.acl == other.acl and (self.cors_config == other.cors_config) and (self.creation_time == other.creation_time) and (self.default_event_based_hold == other.default_event_based_hold) and (self.default_storage_class == other.default_storage_class) and (self.etag == other.etag) and (self.location == other.location) and (self.labels == other.labels) and (self.lifecycle_config == other.lifecycle_config) and (self.location == other.location) and (self.logging_config == other.logging_config) and (self.metadata == other.metadata) and (self.metageneration == other.metageneration) and (self.requester_pays == other.requester_pays) and (self.retention_policy == other.retention_policy) and (self.update_time == other.update_time) and (self.versioning_enabled == other.versioning_enabled) and (self.website_config == other.website_config)

    def is_container(self):
        return True