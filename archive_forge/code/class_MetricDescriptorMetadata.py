from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricDescriptorMetadata(_messages.Message):
    """Additional annotations that can be used to guide the usage of a metric.

  Enums:
    LaunchStageValueValuesEnum: The launch stage of the metric definition.

  Fields:
    ingestDelay: The delay of data points caused by ingestion. Data points
      older than this age are guaranteed to be ingested and available to be
      read, excluding data loss due to errors.
    launchStage: The launch stage of the metric definition.
    samplePeriod: The sampling period of metric data points. For metrics which
      are written periodically, consecutive data points are stored at this
      time interval, excluding data loss due to errors. Metrics with a higher
      granularity have a smaller sampling period.
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """The launch stage of the metric definition.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: Do not use this default value.
      EARLY_ACCESS: Early Access features are limited to a closed group of
        testers. To use these features, you must sign up in advance and sign a
        Trusted Tester agreement (which includes confidentiality provisions).
        These features may be unstable, changed in backward-incompatible ways,
        and are not guaranteed to be released.
      ALPHA: Alpha is a limited availability test for releases before they are
        cleared for widespread use. By Alpha, all significant design issues
        are resolved and we are in the process of verifying functionality.
        Alpha customers need to apply for access, agree to applicable terms,
        and have their projects whitelisted. Alpha releases don't have to be
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
        our [Terms of Service](https://cloud.google.com/terms/) and the
        [Google Cloud Platform Subject to the Deprecation
        Policy](https://cloud.google.com/terms/deprecation) documentation.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        EARLY_ACCESS = 1
        ALPHA = 2
        BETA = 3
        GA = 4
        DEPRECATED = 5
    ingestDelay = _messages.StringField(1)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 2)
    samplePeriod = _messages.StringField(3)