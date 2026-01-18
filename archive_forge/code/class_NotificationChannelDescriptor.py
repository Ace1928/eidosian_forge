from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationChannelDescriptor(_messages.Message):
    """A description of a notification channel. The descriptor includes the
  properties of the channel and the set of labels or fields that must be
  specified to configure channels of a given type.

  Enums:
    LaunchStageValueValuesEnum: The product launch stage for channels of this
      type.
    SupportedTiersValueListEntryValuesEnum:

  Fields:
    description: A human-readable description of the notification channel
      type. The description may include a description of the properties of the
      channel and pointers to external documentation.
    displayName: A human-readable name for the notification channel type. This
      form of the name is suitable for a user interface.
    labels: The set of labels that must be defined to identify a particular
      channel of the corresponding type. Each label includes a description for
      how that field should be populated.
    launchStage: The product launch stage for channels of this type.
    name: The full REST resource name for this descriptor. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/notificationChannelDescriptors/[TYPE] In
      the above, [TYPE] is the value of the type field.
    supportedTiers: The tiers that support this notification channel; the
      project service tier must be one of the supported_tiers.
    type: The type of notification channel, such as "email" and "sms". To view
      the full list of channels, see Channel descriptors
      (https://cloud.google.com/monitoring/alerts/using-channels-api#ncd).
      Notification channel types are globally unique.
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """The product launch stage for channels of this type.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: Do not use this default value.
      UNIMPLEMENTED: The feature is not yet implemented. Users can not use it.
      PRELAUNCH: Prelaunch features are hidden from users and are only visible
        internally.
      EARLY_ACCESS: Early Access features are limited to a closed group of
        testers. To use these features, you must sign up in advance and sign a
        Trusted Tester agreement (which includes confidentiality provisions).
        These features may be unstable, changed in backward-incompatible ways,
        and are not guaranteed to be released.
      ALPHA: Alpha is a limited availability test for releases before they are
        cleared for widespread use. By Alpha, all significant design issues
        are resolved and we are in the process of verifying functionality.
        Alpha customers need to apply for access, agree to applicable terms,
        and have their projects allowlisted. Alpha releases don't have to be
        feature complete, no SLAs are provided, and there are no technical
        support obligations, but they will be far enough along that customers
        can actually use them in test environments or for limited-use tests --
        just like they would in normal production cases.
      BETA: Beta is the point at which we are ready to open a release for any
        customer to use. There are no SLA or technical support obligations in
        a Beta release. Products will be complete from a feature perspective,
        but may have some open outstanding issues. Beta releases are suitable
        for limited production use cases.
      GA: GA features are open to all developers and are considered stable and
        fully qualified for production use.
      DEPRECATED: Deprecated features are scheduled to be shut down and
        removed. For more information, see the "Deprecation Policy" section of
        our Terms of Service (https://cloud.google.com/terms/) and the Google
        Cloud Platform Subject to the Deprecation Policy
        (https://cloud.google.com/terms/deprecation) documentation.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        UNIMPLEMENTED = 1
        PRELAUNCH = 2
        EARLY_ACCESS = 3
        ALPHA = 4
        BETA = 5
        GA = 6
        DEPRECATED = 7

    class SupportedTiersValueListEntryValuesEnum(_messages.Enum):
        """SupportedTiersValueListEntryValuesEnum enum type.

    Values:
      SERVICE_TIER_UNSPECIFIED: An invalid sentinel value, used to indicate
        that a tier has not been provided explicitly.
      SERVICE_TIER_BASIC: The Cloud Monitoring Basic tier, a free tier of
        service that provides basic features, a moderate allotment of logs,
        and access to built-in metrics. A number of features are not available
        in this tier. For more details, see the service tiers documentation
        (https://cloud.google.com/monitoring/workspaces/tiers).
      SERVICE_TIER_PREMIUM: The Cloud Monitoring Premium tier, a higher, more
        expensive tier of service that provides access to all Cloud Monitoring
        features, lets you use Cloud Monitoring with AWS accounts, and has a
        larger allotments for logs and metrics. For more details, see the
        service tiers documentation
        (https://cloud.google.com/monitoring/workspaces/tiers).
    """
        SERVICE_TIER_UNSPECIFIED = 0
        SERVICE_TIER_BASIC = 1
        SERVICE_TIER_PREMIUM = 2
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelDescriptor', 3, repeated=True)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 4)
    name = _messages.StringField(5)
    supportedTiers = _messages.EnumField('SupportedTiersValueListEntryValuesEnum', 6, repeated=True)
    type = _messages.StringField(7)