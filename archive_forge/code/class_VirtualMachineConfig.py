from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VirtualMachineConfig(_messages.Message):
    """The config settings for virtual machine.

  Enums:
    NicTypeValueValuesEnum: Optional. The type of vNIC to be used on this
      interface. This may be gVNIC or VirtioNet.

  Messages:
    GuestAttributesValue: Output only. The Compute Engine guest attributes.
      (see [Project and instance guest
      attributes](https://cloud.google.com/compute/docs/storing-retrieving-
      metadata#guest_attributes)).
    LabelsValue: Optional. The labels to associate with this runtime. Label
      **keys** must contain 1 to 63 characters, and must conform to [RFC
      1035](https://www.ietf.org/rfc/rfc1035.txt). Label **values** may be
      empty, but, if present, must contain 1 to 63 characters, and must
      conform to [RFC 1035](https://www.ietf.org/rfc/rfc1035.txt). No more
      than 32 labels can be associated with a cluster.
    MetadataValue: Optional. The Compute Engine metadata entries to add to
      virtual machine. (see [Project and instance
      metadata](https://cloud.google.com/compute/docs/storing-retrieving-
      metadata#project_and_instance_metadata)).

  Fields:
    acceleratorConfig: Optional. The Compute Engine accelerator configuration
      for this runtime.
    bootImage: Optional. Boot image metadata used for runtime upgradeability.
    containerImages: Optional. Use a list of container images to use as
      Kernels in the notebook instance.
    dataDisk: Required. Data disk option configuration settings.
    encryptionConfig: Optional. Encryption settings for virtual machine data
      disk.
    guestAttributes: Output only. The Compute Engine guest attributes. (see
      [Project and instance guest
      attributes](https://cloud.google.com/compute/docs/storing-retrieving-
      metadata#guest_attributes)).
    internalIpOnly: Optional. If true, runtime will only have internal IP
      addresses. By default, runtimes are not restricted to internal IP
      addresses, and will have ephemeral external IP addresses assigned to
      each vm. This `internal_ip_only` restriction can only be enabled for
      subnetwork enabled networks, and all dependencies must be configured to
      be accessible without external IP addresses.
    labels: Optional. The labels to associate with this runtime. Label
      **keys** must contain 1 to 63 characters, and must conform to [RFC
      1035](https://www.ietf.org/rfc/rfc1035.txt). Label **values** may be
      empty, but, if present, must contain 1 to 63 characters, and must
      conform to [RFC 1035](https://www.ietf.org/rfc/rfc1035.txt). No more
      than 32 labels can be associated with a cluster.
    machineType: Required. The Compute Engine machine type used for runtimes.
      Short name is valid. Examples: * `n1-standard-2` * `e2-standard-8`
    metadata: Optional. The Compute Engine metadata entries to add to virtual
      machine. (see [Project and instance
      metadata](https://cloud.google.com/compute/docs/storing-retrieving-
      metadata#project_and_instance_metadata)).
    network: Optional. The Compute Engine network to be used for machine
      communications. Cannot be specified with subnetwork. If neither
      `network` nor `subnet` is specified, the "default" network of the
      project is used, if it exists. A full URL or partial URI. Examples: * `h
      ttps://www.googleapis.com/compute/v1/projects/[project_id]/global/networ
      ks/default` * `projects/[project_id]/global/networks/default` Runtimes
      are managed resources inside Google Infrastructure. Runtimes support the
      following network configurations: * Google Managed Network (Network &
      subnet are empty) * Consumer Project VPC (network & subnet are
      required). Requires configuring Private Service Access. * Shared VPC
      (network & subnet are required). Requires configuring Private Service
      Access.
    nicType: Optional. The type of vNIC to be used on this interface. This may
      be gVNIC or VirtioNet.
    reservedIpRange: Optional. Reserved IP Range name is used for VPC Peering.
      The subnetwork allocation will use the range *name* if it's assigned.
      Example: managed-notebooks-range-c PEERING_RANGE_NAME_3=managed-
      notebooks-range-c gcloud compute addresses create $PEERING_RANGE_NAME_3
      \\ --global \\ --prefix-length=24 \\ --description="Google Cloud Managed
      Notebooks Range 24 c" \\ --network=$NETWORK \\ --addresses=192.168.0.0 \\
      --purpose=VPC_PEERING Field value will be: `managed-notebooks-range-c`
    shieldedInstanceConfig: Optional. Shielded VM Instance configuration
      settings.
    subnet: Optional. The Compute Engine subnetwork to be used for machine
      communications. Cannot be specified with network. A full URL or partial
      URI are valid. Examples: *
      `https://www.googleapis.com/compute/v1/projects/[project_id]/regions/us-
      east1/subnetworks/sub0` * `projects/[project_id]/regions/us-
      east1/subnetworks/sub0`
    tags: Optional. The Compute Engine tags to add to runtime (see [Tagging
      instances](https://cloud.google.com/compute/docs/label-or-tag-
      resources#tags)).
    zone: Output only. The zone where the virtual machine is located. If using
      regional request, the notebooks service will pick a location in the
      corresponding runtime region. On a get request, zone will always be
      present. Example: * `us-central1-b`
  """

    class NicTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of vNIC to be used on this interface. This may be
    gVNIC or VirtioNet.

    Values:
      UNSPECIFIED_NIC_TYPE: No type specified.
      VIRTIO_NET: VIRTIO
      GVNIC: GVNIC
    """
        UNSPECIFIED_NIC_TYPE = 0
        VIRTIO_NET = 1
        GVNIC = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class GuestAttributesValue(_messages.Message):
        """Output only. The Compute Engine guest attributes. (see [Project and
    instance guest attributes](https://cloud.google.com/compute/docs/storing-
    retrieving-metadata#guest_attributes)).

    Messages:
      AdditionalProperty: An additional property for a GuestAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type GuestAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a GuestAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels to associate with this runtime. Label **keys**
    must contain 1 to 63 characters, and must conform to [RFC
    1035](https://www.ietf.org/rfc/rfc1035.txt). Label **values** may be
    empty, but, if present, must contain 1 to 63 characters, and must conform
    to [RFC 1035](https://www.ietf.org/rfc/rfc1035.txt). No more than 32
    labels can be associated with a cluster.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Optional. The Compute Engine metadata entries to add to virtual
    machine. (see [Project and instance
    metadata](https://cloud.google.com/compute/docs/storing-retrieving-
    metadata#project_and_instance_metadata)).

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    acceleratorConfig = _messages.MessageField('RuntimeAcceleratorConfig', 1)
    bootImage = _messages.MessageField('BootImage', 2)
    containerImages = _messages.MessageField('ContainerImage', 3, repeated=True)
    dataDisk = _messages.MessageField('LocalDisk', 4)
    encryptionConfig = _messages.MessageField('EncryptionConfig', 5)
    guestAttributes = _messages.MessageField('GuestAttributesValue', 6)
    internalIpOnly = _messages.BooleanField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    machineType = _messages.StringField(9)
    metadata = _messages.MessageField('MetadataValue', 10)
    network = _messages.StringField(11)
    nicType = _messages.EnumField('NicTypeValueValuesEnum', 12)
    reservedIpRange = _messages.StringField(13)
    shieldedInstanceConfig = _messages.MessageField('RuntimeShieldedInstanceConfig', 14)
    subnet = _messages.StringField(15)
    tags = _messages.StringField(16, repeated=True)
    zone = _messages.StringField(17)