from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogBucket(_messages.Message):
    """Describes a repository in which log entries are stored.

  Enums:
    LifecycleStateValueValuesEnum: Output only. The bucket lifecycle state.
    UnmetAnalyticsUpgradeRequirementsValueListEntryValuesEnum:

  Fields:
    analyticsEnabled: Whether log analytics is enabled for this bucket.Once
      enabled, log analytics features cannot be disabled.
    analyticsUpgradeTime: Output only. The time that the bucket was upgraded
      to enable analytics. This will eventually be deprecated once there is
      not a need to upgrade existing buckets (i.e. when analytics becomes
      default-enabled).
    cmekSettings: Optional. The CMEK settings of the log bucket. If present,
      new log entries written to this log bucket are encrypted using the CMEK
      key provided in this configuration. If a log bucket has CMEK settings,
      the CMEK settings cannot be disabled later by updating the log bucket.
      Changing the KMS key is allowed.
    createTime: Output only. The creation timestamp of the bucket. This is not
      set for any of the default buckets.
    description: Optional. Describes this bucket.
    indexConfigs: Optional. A list of indexed fields and related configuration
      data.
    lifecycleState: Output only. The bucket lifecycle state.
    locked: Optional. Whether the bucket is locked.The retention period on a
      locked bucket cannot be changed. Locked buckets may only be deleted if
      they are empty.
    name: Output only. The resource name of the bucket.For
      example:projects/my-project/locations/global/buckets/my-bucketFor a list
      of supported locations, see Supported Regions
      (https://cloud.google.com/logging/docs/region-support)For the location
      of global it is unspecified where log entries are actually stored.After
      a bucket has been created, the location cannot be changed.
    restrictedFields: Optional. Log entry field paths that are denied access
      in this bucket.The following fields and their children are eligible:
      textPayload, jsonPayload, protoPayload, httpRequest, labels,
      sourceLocation.Restricting a repeated field will restrict all values.
      Adding a parent will block all child fields. (e.g. foo.bar will block
      foo.bar.baz)
    retentionDays: Optional. Logs will be retained by default for this amount
      of time, after which they will automatically be deleted. The minimum
      retention period is 1 day. If this value is set to zero at bucket
      creation time, the default time of 30 days will be used.
    unmetAnalyticsUpgradeRequirements: Output only. The requirements for an
      upgrade to analytics that are not satisfied by the current bucket
      configuration, in an arbitrary order. This will eventually be deprecated
      once there is not a need to upgrade existing buckets (i.e. when
      analytics becomes default-enabled).
    updateTime: Output only. The last update timestamp of the bucket.
  """

    class LifecycleStateValueValuesEnum(_messages.Enum):
        """Output only. The bucket lifecycle state.

    Values:
      LIFECYCLE_STATE_UNSPECIFIED: Unspecified state. This is only used/useful
        for distinguishing unset values.
      ACTIVE: The normal and active state.
      DELETE_REQUESTED: The resource has been marked for deletion by the user.
        For some resources (e.g. buckets), this can be reversed by an un-
        delete operation.
      UPDATING: The resource has been marked for an update by the user. It
        will remain in this state until the update is complete.
      CREATING: The resource has been marked for creation by the user. It will
        remain in this state until the creation is complete.
      FAILED: The resource is in an INTERNAL error state.
      ARCHIVED: The resource has been archived such that it can still be
        queried but can no longer be modified or used as a sink destination.
        The source bucket after a move bucket operation enters this state.
    """
        LIFECYCLE_STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETE_REQUESTED = 2
        UPDATING = 3
        CREATING = 4
        FAILED = 5
        ARCHIVED = 6

    class UnmetAnalyticsUpgradeRequirementsValueListEntryValuesEnum(_messages.Enum):
        """UnmetAnalyticsUpgradeRequirementsValueListEntryValuesEnum enum type.

    Values:
      REQUIREMENT_UNSPECIFIED: Unexpected default.
      ACTIVE_LIFECYCLE_STATE: The requirement that a bucket must be in the
        ACTIVE lifecycle state.
      GLOBAL_BUCKET_REGION: The requirement that a bucket must be in the
        "global" region. This requirement is deprecated and replaced with
        SUPPORTED_BUCKET_REGION.
      DEFAULT_RETENTION_DURATION: The requirement that buckets other than the
        "_Required" bucket must have the default retention duration of 30 days
        set. This requirement is deprecated as buckets with custom retention
        can now upgrade to Log Analytics.
      REQUIRED_RETENTION_DURATION: The requirement that the "_Required" bucket
        must have its default retention of 400 days set.
      FIELD_LEVEL_ACCESS_CONTROLS_UNSET: The requirement that no field level
        access controls are configured for the bucket. This requirement is
        deprecated as buckets with restricted field ACLs can now be upgraded
        to Log Analytics. However, the following applies: 1. Users who do not
        have access to the restricted fields will not be able to query any
        views in the bucket using Log Analytics. 2. Users who have access to
        all restricted fields can query any views they have access to in the
        bucket using Log Analytics. 3. If a linked dataset exists in the
        bucket, all data accessible via views in the bucket is queryable via
        the linked dataset in BigQuery. Field level ACLs should be applied to
        linked datasets using BigQuery access control mechanisms.
      CMEK_UNSET: The requirement that no CMEK configuration is set for the
        bucket. This requirement is deprecated as buckets with CMEK can now be
        upgraded to Log Analytics.
      NOT_LOCKED: The requirement that the bucket is not locked.
      ORGANIZATION_BUCKET: The requirement that the bucket must not be
        contained within an org.
      FOLDER_BUCKET: The requirement that the bucket must not be contained
        within a folder.
      BILLING_ACCOUNT_BUCKET: The requirement that the bucket must not be
        contained within a billing account.
      SUPPORTED_BUCKET_REGION: The requirement that the bucket must be in a
        region supported by Log Analytics.
    """
        REQUIREMENT_UNSPECIFIED = 0
        ACTIVE_LIFECYCLE_STATE = 1
        GLOBAL_BUCKET_REGION = 2
        DEFAULT_RETENTION_DURATION = 3
        REQUIRED_RETENTION_DURATION = 4
        FIELD_LEVEL_ACCESS_CONTROLS_UNSET = 5
        CMEK_UNSET = 6
        NOT_LOCKED = 7
        ORGANIZATION_BUCKET = 8
        FOLDER_BUCKET = 9
        BILLING_ACCOUNT_BUCKET = 10
        SUPPORTED_BUCKET_REGION = 11
    analyticsEnabled = _messages.BooleanField(1)
    analyticsUpgradeTime = _messages.StringField(2)
    cmekSettings = _messages.MessageField('CmekSettings', 3)
    createTime = _messages.StringField(4)
    description = _messages.StringField(5)
    indexConfigs = _messages.MessageField('IndexConfig', 6, repeated=True)
    lifecycleState = _messages.EnumField('LifecycleStateValueValuesEnum', 7)
    locked = _messages.BooleanField(8)
    name = _messages.StringField(9)
    restrictedFields = _messages.StringField(10, repeated=True)
    retentionDays = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    unmetAnalyticsUpgradeRequirements = _messages.EnumField('UnmetAnalyticsUpgradeRequirementsValueListEntryValuesEnum', 12, repeated=True)
    updateTime = _messages.StringField(13)