from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeEnvironment(_messages.Message):
    """The environment values to set at runtime. LINT.IfChange

  Enums:
    IpConfigurationValueValuesEnum: Optional. Configuration for VM IPs.
    StreamingModeValueValuesEnum: Optional. Specifies the Streaming Engine
      message processing guarantees. Reduces cost and latency but might result
      in duplicate messages committed to storage. Designed to run simple
      mapping streaming ETL jobs at the lowest cost. For example, Change Data
      Capture (CDC) to BigQuery is a canonical use case. For more information,
      see [Set the pipeline streaming
      mode](https://cloud.google.com/dataflow/docs/guides/streaming-modes).

  Messages:
    AdditionalUserLabelsValue: Optional. Additional user labels to be
      specified for the job. Keys and values should follow the restrictions
      specified in the [labeling
      restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions) page. An object containing a list of "key":
      value pairs. Example: { "name": "wrench", "mass": "1kg", "count": "3" }.

  Fields:
    additionalExperiments: Optional. Additional experiment flags for the job,
      specified with the `--experiments` option.
    additionalUserLabels: Optional. Additional user labels to be specified for
      the job. Keys and values should follow the restrictions specified in the
      [labeling restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions) page. An object containing a list of "key":
      value pairs. Example: { "name": "wrench", "mass": "1kg", "count": "3" }.
    bypassTempDirValidation: Optional. Whether to bypass the safety checks for
      the job's temporary directory. Use with caution.
    diskSizeGb: Optional. The disk size, in gigabytes, to use on each remote
      Compute Engine worker instance.
    enableStreamingEngine: Optional. Whether to enable Streaming Engine for
      the job.
    ipConfiguration: Optional. Configuration for VM IPs.
    kmsKeyName: Optional. Name for the Cloud KMS key for the job. Key format
      is: projects//locations//keyRings//cryptoKeys/
    machineType: Optional. The machine type to use for the job. Defaults to
      the value from the template if not specified.
    maxWorkers: Optional. The maximum number of Google Compute Engine
      instances to be made available to your pipeline during execution, from 1
      to 1000. The default value is 1.
    network: Optional. Network to which VMs will be assigned. If empty or
      unspecified, the service will use the network "default".
    numWorkers: Optional. The initial number of Google Compute Engine
      instances for the job. The default value is 11.
    serviceAccountEmail: Optional. The email address of the service account to
      run the job as.
    streamingMode: Optional. Specifies the Streaming Engine message processing
      guarantees. Reduces cost and latency but might result in duplicate
      messages committed to storage. Designed to run simple mapping streaming
      ETL jobs at the lowest cost. For example, Change Data Capture (CDC) to
      BigQuery is a canonical use case. For more information, see [Set the
      pipeline streaming
      mode](https://cloud.google.com/dataflow/docs/guides/streaming-modes).
    subnetwork: Optional. Subnetwork to which VMs will be assigned, if
      desired. You can specify a subnetwork using either a complete URL or an
      abbreviated path. Expected to be of the form "https://www.googleapis.com
      /compute/v1/projects/HOST_PROJECT_ID/regions/REGION/subnetworks/SUBNETWO
      RK" or "regions/REGION/subnetworks/SUBNETWORK". If the subnetwork is
      located in a Shared VPC network, you must use the complete URL.
    tempLocation: Required. The Cloud Storage path to use for temporary files.
      Must be a valid Cloud Storage URL, beginning with `gs://`.
    workerRegion: Required. The Compute Engine region
      (https://cloud.google.com/compute/docs/regions-zones/regions-zones) in
      which worker processing should occur, e.g. "us-west1". Mutually
      exclusive with worker_zone. If neither worker_region nor worker_zone is
      specified, default to the control plane's region.
    workerZone: Optional. The Compute Engine zone
      (https://cloud.google.com/compute/docs/regions-zones/regions-zones) in
      which worker processing should occur, e.g. "us-west1-a". Mutually
      exclusive with worker_region. If neither worker_region nor worker_zone
      is specified, a zone in the control plane's region is chosen based on
      available capacity. If both `worker_zone` and `zone` are set,
      `worker_zone` takes precedence.
    zone: Optional. The Compute Engine [availability
      zone](https://cloud.google.com/compute/docs/regions-zones/regions-zones)
      for launching worker instances to run your pipeline. In the future,
      worker_zone will take precedence.
  """

    class IpConfigurationValueValuesEnum(_messages.Enum):
        """Optional. Configuration for VM IPs.

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
        """Optional. Additional user labels to be specified for the job. Keys and
    values should follow the restrictions specified in the [labeling
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
    bypassTempDirValidation = _messages.BooleanField(3)
    diskSizeGb = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    enableStreamingEngine = _messages.BooleanField(5)
    ipConfiguration = _messages.EnumField('IpConfigurationValueValuesEnum', 6)
    kmsKeyName = _messages.StringField(7)
    machineType = _messages.StringField(8)
    maxWorkers = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    network = _messages.StringField(10)
    numWorkers = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    serviceAccountEmail = _messages.StringField(12)
    streamingMode = _messages.EnumField('StreamingModeValueValuesEnum', 13)
    subnetwork = _messages.StringField(14)
    tempLocation = _messages.StringField(15)
    workerRegion = _messages.StringField(16)
    workerZone = _messages.StringField(17)
    zone = _messages.StringField(18)