from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlexTemplateRuntimeEnvironment(_messages.Message):
    """The environment values to be set at runtime for flex template.
  LINT.IfChange

  Enums:
    AutoscalingAlgorithmValueValuesEnum: The algorithm to use for autoscaling
    FlexrsGoalValueValuesEnum: Set FlexRS goal for the job.
      https://cloud.google.com/dataflow/docs/guides/flexrs
    IpConfigurationValueValuesEnum: Configuration for VM IPs.
    StreamingModeValueValuesEnum: Optional. Specifies the Streaming Engine
      message processing guarantees. Reduces cost and latency but might result
      in duplicate messages committed to storage. Designed to run simple
      mapping streaming ETL jobs at the lowest cost. For example, Change Data
      Capture (CDC) to BigQuery is a canonical use case. For more information,
      see [Set the pipeline streaming
      mode](https://cloud.google.com/dataflow/docs/guides/streaming-modes).

  Messages:
    AdditionalUserLabelsValue: Additional user labels to be specified for the
      job. Keys and values must follow the restrictions specified in the
      [labeling restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions) page. An object containing a list of "key":
      value pairs. Example: { "name": "wrench", "mass": "1kg", "count": "3" }.

  Fields:
    additionalExperiments: Additional experiment flags for the job.
    additionalUserLabels: Additional user labels to be specified for the job.
      Keys and values must follow the restrictions specified in the [labeling
      restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions) page. An object containing a list of "key":
      value pairs. Example: { "name": "wrench", "mass": "1kg", "count": "3" }.
    autoscalingAlgorithm: The algorithm to use for autoscaling
    diskSizeGb: Worker disk size, in gigabytes.
    dumpHeapOnOom: If true, when processing time is spent almost entirely on
      garbage collection (GC), saves a heap dump before ending the thread or
      process. If false, ends the thread or process without saving a heap
      dump. Does not save a heap dump when the Java Virtual Machine (JVM) has
      an out of memory error during processing. The location of the heap file
      is either echoed back to the user, or the user is given the opportunity
      to download the heap file.
    enableLauncherVmSerialPortLogging: If true serial port logging will be
      enabled for the launcher VM.
    enableStreamingEngine: Whether to enable Streaming Engine for the job.
    flexrsGoal: Set FlexRS goal for the job.
      https://cloud.google.com/dataflow/docs/guides/flexrs
    ipConfiguration: Configuration for VM IPs.
    kmsKeyName: Name for the Cloud KMS key for the job. Key format is:
      projects//locations//keyRings//cryptoKeys/
    launcherMachineType: The machine type to use for launching the job. The
      default is n1-standard-1.
    machineType: The machine type to use for the job. Defaults to the value
      from the template if not specified.
    maxWorkers: The maximum number of Google Compute Engine instances to be
      made available to your pipeline during execution, from 1 to 1000.
    network: Network to which VMs will be assigned. If empty or unspecified,
      the service will use the network "default".
    numWorkers: The initial number of Google Compute Engine instances for the
      job.
    saveHeapDumpsToGcsPath: Cloud Storage bucket (directory) to upload heap
      dumps to. Enabling this field implies that `dump_heap_on_oom` is set to
      true.
    sdkContainerImage: Docker registry location of container image to use for
      the 'worker harness. Default is the container for the version of the
      SDK. Note this field is only valid for portable pipelines.
    serviceAccountEmail: The email address of the service account to run the
      job as.
    stagingLocation: The Cloud Storage path for staging local files. Must be a
      valid Cloud Storage URL, beginning with `gs://`.
    streamingMode: Optional. Specifies the Streaming Engine message processing
      guarantees. Reduces cost and latency but might result in duplicate
      messages committed to storage. Designed to run simple mapping streaming
      ETL jobs at the lowest cost. For example, Change Data Capture (CDC) to
      BigQuery is a canonical use case. For more information, see [Set the
      pipeline streaming
      mode](https://cloud.google.com/dataflow/docs/guides/streaming-modes).
    subnetwork: Subnetwork to which VMs will be assigned, if desired. You can
      specify a subnetwork using either a complete URL or an abbreviated path.
      Expected to be of the form "https://www.googleapis.com/compute/v1/projec
      ts/HOST_PROJECT_ID/regions/REGION/subnetworks/SUBNETWORK" or
      "regions/REGION/subnetworks/SUBNETWORK". If the subnetwork is located in
      a Shared VPC network, you must use the complete URL.
    tempLocation: The Cloud Storage path to use for temporary files. Must be a
      valid Cloud Storage URL, beginning with `gs://`.
    workerRegion: The Compute Engine region
      (https://cloud.google.com/compute/docs/regions-zones/regions-zones) in
      which worker processing should occur, e.g. "us-west1". Mutually
      exclusive with worker_zone. If neither worker_region nor worker_zone is
      specified, default to the control plane's region.
    workerZone: The Compute Engine zone
      (https://cloud.google.com/compute/docs/regions-zones/regions-zones) in
      which worker processing should occur, e.g. "us-west1-a". Mutually
      exclusive with worker_region. If neither worker_region nor worker_zone
      is specified, a zone in the control plane's region is chosen based on
      available capacity. If both `worker_zone` and `zone` are set,
      `worker_zone` takes precedence.
    zone: The Compute Engine [availability
      zone](https://cloud.google.com/compute/docs/regions-zones/regions-zones)
      for launching worker instances to run your pipeline. In the future,
      worker_zone will take precedence.
  """

    class AutoscalingAlgorithmValueValuesEnum(_messages.Enum):
        """The algorithm to use for autoscaling

    Values:
      AUTOSCALING_ALGORITHM_UNKNOWN: The algorithm is unknown, or unspecified.
      AUTOSCALING_ALGORITHM_NONE: Disable autoscaling.
      AUTOSCALING_ALGORITHM_BASIC: Increase worker count over time to reduce
        job execution time.
    """
        AUTOSCALING_ALGORITHM_UNKNOWN = 0
        AUTOSCALING_ALGORITHM_NONE = 1
        AUTOSCALING_ALGORITHM_BASIC = 2

    class FlexrsGoalValueValuesEnum(_messages.Enum):
        """Set FlexRS goal for the job.
    https://cloud.google.com/dataflow/docs/guides/flexrs

    Values:
      FLEXRS_UNSPECIFIED: Run in the default mode.
      FLEXRS_SPEED_OPTIMIZED: Optimize for lower execution time.
      FLEXRS_COST_OPTIMIZED: Optimize for lower cost.
    """
        FLEXRS_UNSPECIFIED = 0
        FLEXRS_SPEED_OPTIMIZED = 1
        FLEXRS_COST_OPTIMIZED = 2

    class IpConfigurationValueValuesEnum(_messages.Enum):
        """Configuration for VM IPs.

    Values:
      WORKER_IP_UNSPECIFIED: The configuration is unknown, or unspecified.
      WORKER_IP_PUBLIC: Workers should have public IP addresses.
      WORKER_IP_PRIVATE: Workers should have private IP addresses.
    """
        WORKER_IP_UNSPECIFIED = 0
        WORKER_IP_PUBLIC = 1
        WORKER_IP_PRIVATE = 2

    class StreamingModeValueValuesEnum(_messages.Enum):
        """Optional. Specifies the Streaming Engine message processing
    guarantees. Reduces cost and latency but might result in duplicate
    messages committed to storage. Designed to run simple mapping streaming
    ETL jobs at the lowest cost. For example, Change Data Capture (CDC) to
    BigQuery is a canonical use case. For more information, see [Set the
    pipeline streaming
    mode](https://cloud.google.com/dataflow/docs/guides/streaming-modes).

    Values:
      STREAMING_MODE_UNSPECIFIED: Run in the default mode.
      STREAMING_MODE_EXACTLY_ONCE: In this mode, message deduplication is
        performed against persistent state to make sure each message is
        processed and committed to storage exactly once.
      STREAMING_MODE_AT_LEAST_ONCE: Message deduplication is not performed.
        Messages might be processed multiple times, and the results are
        applied multiple times. Note: Setting this value also enables
        Streaming Engine and Streaming Engine resource-based billing.
    """
        STREAMING_MODE_UNSPECIFIED = 0
        STREAMING_MODE_EXACTLY_ONCE = 1
        STREAMING_MODE_AT_LEAST_ONCE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AdditionalUserLabelsValue(_messages.Message):
        """Additional user labels to be specified for the job. Keys and values
    must follow the restrictions specified in the [labeling
    restrictions](https://cloud.google.com/compute/docs/labeling-
    resources#restrictions) page. An object containing a list of "key": value
    pairs. Example: { "name": "wrench", "mass": "1kg", "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a
        AdditionalUserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AdditionalUserLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AdditionalUserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    additionalExperiments = _messages.StringField(1, repeated=True)
    additionalUserLabels = _messages.MessageField('AdditionalUserLabelsValue', 2)
    autoscalingAlgorithm = _messages.EnumField('AutoscalingAlgorithmValueValuesEnum', 3)
    diskSizeGb = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    dumpHeapOnOom = _messages.BooleanField(5)
    enableLauncherVmSerialPortLogging = _messages.BooleanField(6)
    enableStreamingEngine = _messages.BooleanField(7)
    flexrsGoal = _messages.EnumField('FlexrsGoalValueValuesEnum', 8)
    ipConfiguration = _messages.EnumField('IpConfigurationValueValuesEnum', 9)
    kmsKeyName = _messages.StringField(10)
    launcherMachineType = _messages.StringField(11)
    machineType = _messages.StringField(12)
    maxWorkers = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    network = _messages.StringField(14)
    numWorkers = _messages.IntegerField(15, variant=_messages.Variant.INT32)
    saveHeapDumpsToGcsPath = _messages.StringField(16)
    sdkContainerImage = _messages.StringField(17)
    serviceAccountEmail = _messages.StringField(18)
    stagingLocation = _messages.StringField(19)
    streamingMode = _messages.EnumField('StreamingModeValueValuesEnum', 20)
    subnetwork = _messages.StringField(21)
    tempLocation = _messages.StringField(22)
    workerRegion = _messages.StringField(23)
    workerZone = _messages.StringField(24)
    zone = _messages.StringField(25)