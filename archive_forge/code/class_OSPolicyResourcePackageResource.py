from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResource(_messages.Message):
    """A resource that manages a system package.

  Enums:
    DesiredStateValueValuesEnum: Required. The desired state the agent should
      maintain for this package.

  Fields:
    apt: A package managed by Apt.
    deb: A deb package file.
    desiredState: Required. The desired state the agent should maintain for
      this package.
    googet: A package managed by GooGet.
    msi: An MSI package.
    rpm: An rpm package file.
    yum: A package managed by YUM.
    zypper: A package managed by Zypper.
  """

    class DesiredStateValueValuesEnum(_messages.Enum):
        """Required. The desired state the agent should maintain for this
    package.

    Values:
      DESIRED_STATE_UNSPECIFIED: Unspecified is invalid.
      INSTALLED: Ensure that the package is installed.
      REMOVED: The agent ensures that the package is not installed and
        uninstalls it if detected.
    """
        DESIRED_STATE_UNSPECIFIED = 0
        INSTALLED = 1
        REMOVED = 2
    apt = _messages.MessageField('OSPolicyResourcePackageResourceAPT', 1)
    deb = _messages.MessageField('OSPolicyResourcePackageResourceDeb', 2)
    desiredState = _messages.EnumField('DesiredStateValueValuesEnum', 3)
    googet = _messages.MessageField('OSPolicyResourcePackageResourceGooGet', 4)
    msi = _messages.MessageField('OSPolicyResourcePackageResourceMSI', 5)
    rpm = _messages.MessageField('OSPolicyResourcePackageResourceRPM', 6)
    yum = _messages.MessageField('OSPolicyResourcePackageResourceYUM', 7)
    zypper = _messages.MessageField('OSPolicyResourcePackageResourceZypper', 8)