from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GceSetup(_messages.Message):
    """The definition of how to configure a VM instance outside of Resources
  and Identity.

  Messages:
    MetadataValue: Optional. Custom metadata to apply to this instance.

  Fields:
    acceleratorConfigs: Optional. The hardware accelerators used on this
      instance. If you use accelerators, make sure that your configuration has
      [enough vCPUs and memory to support the `machine_type` you have
      selected](https://cloud.google.com/compute/docs/gpus/#gpus-list).
      Currently supports only one accelerator configuration.
    bootDisk: Optional. The boot disk for the VM.
    containerImage: Optional. Use a container image to start the notebook
      instance.
    dataDisks: Optional. Data disks attached to the VM instance. Currently
      supports only one data disk.
    disablePublicIp: Optional. If true, no external IP will be assigned to
      this VM instance.
    enableIpForwarding: Optional. Flag to enable ip forwarding or not, default
      false/off. https://cloud.google.com/vpc/docs/using-routes#canipforward
    gpuDriverConfig: Optional. Configuration for GPU drivers.
    machineType: Optional. The machine type of the VM instance.
      https://cloud.google.com/compute/docs/machine-resource
    metadata: Optional. Custom metadata to apply to this instance.
    networkInterfaces: Optional. The network interfaces for the VM. Supports
      only one interface.
    serviceAccounts: Optional. The service account that serves as an identity
      for the VM instance. Currently supports only one service account.
    shieldedInstanceConfig: Optional. Shielded VM configuration. [Images using
      supported Shielded VM
      features](https://cloud.google.com/compute/docs/instances/modifying-
      shielded-vm).
    tags: Optional. The Compute Engine tags to add to runtime (see [Tagging
      instances](https://cloud.google.com/compute/docs/label-or-tag-
      resources#tags)).
    vmImage: Optional. Use a Compute Engine VM image to start the notebook
      instance.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Optional. Custom metadata to apply to this instance.

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
    acceleratorConfigs = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    bootDisk = _messages.MessageField('BootDisk', 2)
    containerImage = _messages.MessageField('ContainerImage', 3)
    dataDisks = _messages.MessageField('DataDisk', 4, repeated=True)
    disablePublicIp = _messages.BooleanField(5)
    enableIpForwarding = _messages.BooleanField(6)
    gpuDriverConfig = _messages.MessageField('GPUDriverConfig', 7)
    machineType = _messages.StringField(8)
    metadata = _messages.MessageField('MetadataValue', 9)
    networkInterfaces = _messages.MessageField('NetworkInterface', 10, repeated=True)
    serviceAccounts = _messages.MessageField('ServiceAccount', 11, repeated=True)
    shieldedInstanceConfig = _messages.MessageField('ShieldedInstanceConfig', 12)
    tags = _messages.StringField(13, repeated=True)
    vmImage = _messages.MessageField('VmImage', 14)