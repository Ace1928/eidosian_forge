from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeGuestOsFeature(_messages.Message):
    """Optional. A list of features to enable on the guest operating system.
  Applicable only for bootable images. Read [Enabling guest operating system
  features](https://cloud.google.com/compute/docs/images/create-delete-
  deprecate-private-images#guest-os-features) to see a list of available
  options. Guest OS features for boot disk.

  Fields:
    type: The ID of a supported feature. Read [Enabling guest operating system
      features](https://cloud.google.com/compute/docs/images/create-delete-
      deprecate-private-images#guest-os-features) to see a list of available
      options. Valid values: * `FEATURE_TYPE_UNSPECIFIED` * `MULTI_IP_SUBNET`
      * `SECURE_BOOT` * `UEFI_COMPATIBLE` * `VIRTIO_SCSI_MULTIQUEUE` *
      `WINDOWS`
  """
    type = _messages.StringField(1)