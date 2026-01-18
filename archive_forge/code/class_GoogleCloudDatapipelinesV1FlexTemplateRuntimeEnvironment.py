from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment(_messages.Message):
    """The environment values to be set at runtime for a Flex Template.

  Enums:
    FlexrsGoalValueValuesEnum: Set FlexRS goal for the job.
      https://cloud.google.com/dataflow/docs/guides/flexrs
    IpConfigurationValueValuesEnum: Configuration for VM IPs.

  Messages:
    AdditionalUserLabelsValue: Additional user labels to be specified for the
      job. Keys and values must follow the restrictions specified in the
      [labeling restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions). An object containing a list of key/value pairs.
      Example: `{ "name": "wrench", "mass": "1kg", "count": "3" }`.

  Fields:
    additionalExperiments: Additional experiment flags for the job.
    additionalUserLabels: Additional user labels to be specified for the job.
      Keys and values must follow the restrictions specified in the [labeling
      restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions). An object containing a list of key/value pairs.
      Example: `{ "name": "wrench", "mass": "1kg", "count": "3" }`.
    enableStreamingEngine: Whether to enable Streaming Engine for the job.
    flexrsGoal: Set FlexRS goal for the job.
      https://cloud.google.com/dataflow/docs/guides/flexrs
    ipConfiguration: Configuration for VM IPs.
    kmsKeyName: Name for the Cloud KMS key for the job. Key format is:
      projects//locations//keyRings//cryptoKeys/
    machineType: The machine type to use for the job. Defaults to the value
      from the template if not specified.
    maxWorkers: The maximum number of Compute Engine instances to be made
      available to your pipeline during execution, from 1 to 1000.
    network: Network to which VMs will be assigned. If empty or unspecified,
      the service will use the network "default".
    numWorkers: The initial number of Compute Engine instances for the job.
    serviceAccountEmail: The email address of the service account to run the
      job as.
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
      specified, defaults to the control plane region.
    workerZone: The Compute Engine zone
      (https://cloud.google.com/compute/docs/regions-zones/regions-zones) in
      which worker processing should occur, e.g. "us-west1-a". Mutually
      exclusive with worker_region. If neither worker_region nor worker_zone
      is specified, a zone in the control plane region is chosen based on
      available capacity. If both `worker_zone` and `zone` are set,
      `worker_zone` takes precedence.
    zone: The Compute Engine [availability
      zone](https://cloud.google.com/compute/docs/regions-zones/regions-zones)
      for launching worker instances to run your pipeline. In the future,
      worker_zone will take precedence.
  """

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AdditionalUserLabelsValue(_messages.Message):
        """Additional user labels to be specified for the job. Keys and values
    must follow the restrictions specified in the [labeling
    restrictions](https://cloud.google.com/compute/docs/labeling-
    resources#restrictions). An object containing a list of key/value pairs.
    Example: `{ "name": "wrench", "mass": "1kg", "count": "3" }`.

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
    enableStreamingEngine = _messages.BooleanField(3)
    flexrsGoal = _messages.EnumField('FlexrsGoalValueValuesEnum', 4)
    ipConfiguration = _messages.EnumField('IpConfigurationValueValuesEnum', 5)
    kmsKeyName = _messages.StringField(6)
    machineType = _messages.StringField(7)
    maxWorkers = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    network = _messages.StringField(9)
    numWorkers = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    serviceAccountEmail = _messages.StringField(11)
    subnetwork = _messages.StringField(12)
    tempLocation = _messages.StringField(13)
    workerRegion = _messages.StringField(14)
    workerZone = _messages.StringField(15)
    zone = _messages.StringField(16)