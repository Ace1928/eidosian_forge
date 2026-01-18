from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RebootConfigValueValuesEnum(_messages.Enum):
    """Post-patch reboot settings.

    Values:
      REBOOT_CONFIG_UNSPECIFIED: The default behavior is DEFAULT.
      DEFAULT: The agent decides if a reboot is necessary by checking signals
        such as registry keys on Windows or `/var/run/reboot-required` on APT
        based systems. On RPM based systems, a set of core system package
        install times are compared with system boot time.
      ALWAYS: Always reboot the machine after the update completes.
      NEVER: Never reboot the machine after the update completes.
    """
    REBOOT_CONFIG_UNSPECIFIED = 0
    DEFAULT = 1
    ALWAYS = 2
    NEVER = 3