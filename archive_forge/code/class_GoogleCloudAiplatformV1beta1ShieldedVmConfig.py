from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ShieldedVmConfig(_messages.Message):
    """A set of Shielded Instance options. See [Images using supported Shielded
  VM features](https://cloud.google.com/compute/docs/instances/modifying-
  shielded-vm).

  Fields:
    enableSecureBoot: Defines whether the instance has [Secure
      Boot](https://cloud.google.com/compute/shielded-vm/docs/shielded-
      vm#secure-boot) enabled. Secure Boot helps ensure that the system only
      runs authentic software by verifying the digital signature of all boot
      components, and halting the boot process if signature verification
      fails.
  """
    enableSecureBoot = _messages.BooleanField(1)