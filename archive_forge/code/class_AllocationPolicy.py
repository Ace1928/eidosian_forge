from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationPolicy(_messages.Message):
    """A Job's resource allocation policy describes when, where, and how
  compute resources should be allocated for the Job.

  Enums:
    ProvisioningModelsValueListEntryValuesEnum:

  Messages:
    LabelsValue: Labels applied to all VM instances and other resources
      created by AllocationPolicy. Labels could be user provided or system
      generated. You can assign up to 64 labels. [Google Compute Engine label
      restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions) apply. Label names that start with "goog-" or
      "google-" are reserved.

  Fields:
    instance: Deprecated: please use instances[0].policy instead.
    instanceTemplates: Deprecated: please use instances[0].template instead.
    instances: Describe instances that can be created by this
      AllocationPolicy. Only instances[0] is supported now.
    labels: Labels applied to all VM instances and other resources created by
      AllocationPolicy. Labels could be user provided or system generated. You
      can assign up to 64 labels. [Google Compute Engine label
      restrictions](https://cloud.google.com/compute/docs/labeling-
      resources#restrictions) apply. Label names that start with "goog-" or
      "google-" are reserved.
    location: Location where compute resources should be allocated for the
      Job.
    network: The network policy. If you define an instance template in the
      `InstancePolicyOrTemplate` field, Batch will use the network settings in
      the instance template instead of this field.
    placement: The placement policy.
    provisioningModels: Deprecated: please use
      instances[0].policy.provisioning_model instead.
    serviceAccount: Defines the service account for Batch-created VMs. If
      omitted, the [default Compute Engine service
      account](https://cloud.google.com/compute/docs/access/service-
      accounts#default_service_account) is used. Must match the service
      account specified in any used instance template configured in the Batch
      job. Includes the following fields: * email: The service account's email
      address. If not set, the default Compute Engine service account is used.
      * scopes: Additional OAuth scopes to grant the service account, beyond
      the default cloud-platform scope. (list of strings)
    serviceAccountEmail: Deprecated: please use service_account instead.
    tags: Optional. Tags applied to the VM instances. The tags identify valid
      sources or targets for network firewalls. Each tag must be 1-63
      characters long, and comply with
      [RFC1035](https://www.ietf.org/rfc/rfc1035.txt).
  """

    class ProvisioningModelsValueListEntryValuesEnum(_messages.Enum):
        """ProvisioningModelsValueListEntryValuesEnum enum type.

    Values:
      PROVISIONING_MODEL_UNSPECIFIED: Unspecified.
      STANDARD: Standard VM.
      SPOT: SPOT VM.
      PREEMPTIBLE: Preemptible VM (PVM). Above SPOT VM is the preferable model
        for preemptible VM instances: the old preemptible VM model (indicated
        by this field) is the older model, and has been migrated to use the
        SPOT model as the underlying technology. This old model will still be
        supported.
    """
        PROVISIONING_MODEL_UNSPECIFIED = 0
        STANDARD = 1
        SPOT = 2
        PREEMPTIBLE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels applied to all VM instances and other resources created by
    AllocationPolicy. Labels could be user provided or system generated. You
    can assign up to 64 labels. [Google Compute Engine label
    restrictions](https://cloud.google.com/compute/docs/labeling-
    resources#restrictions) apply. Label names that start with "goog-" or
    "google-" are reserved.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    instance = _messages.MessageField('InstancePolicy', 1)
    instanceTemplates = _messages.StringField(2, repeated=True)
    instances = _messages.MessageField('InstancePolicyOrTemplate', 3, repeated=True)
    labels = _messages.MessageField('LabelsValue', 4)
    location = _messages.MessageField('LocationPolicy', 5)
    network = _messages.MessageField('NetworkPolicy', 6)
    placement = _messages.MessageField('PlacementPolicy', 7)
    provisioningModels = _messages.EnumField('ProvisioningModelsValueListEntryValuesEnum', 8, repeated=True)
    serviceAccount = _messages.MessageField('ServiceAccount', 9)
    serviceAccountEmail = _messages.StringField(10)
    tags = _messages.StringField(11, repeated=True)