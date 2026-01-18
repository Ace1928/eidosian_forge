from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchConfig(_messages.Message):
    """Patch configuration specifications. Contains details on how to apply the
  patch(es) to a VM instance.

  Enums:
    RebootConfigValueValuesEnum: Post-patch reboot settings.

  Fields:
    apt: Apt update settings. Use this setting to override the default `apt`
      patch rules.
    goo: Goo update settings. Use this setting to override the default `goo`
      patch rules.
    migInstancesAllowed: Allows the patch job to run on Managed instance
      groups (MIGs).
    postStep: The `ExecStep` to run after the patch update.
    preStep: The `ExecStep` to run before the patch update.
    rebootConfig: Post-patch reboot settings.
    windowsUpdate: Windows update settings. Use this override the default
      windows patch rules.
    yum: Yum update settings. Use this setting to override the default `yum`
      patch rules.
    zypper: Zypper update settings. Use this setting to override the default
      `zypper` patch rules.
  """

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
    apt = _messages.MessageField('AptSettings', 1)
    goo = _messages.MessageField('GooSettings', 2)
    migInstancesAllowed = _messages.BooleanField(3)
    postStep = _messages.MessageField('ExecStep', 4)
    preStep = _messages.MessageField('ExecStep', 5)
    rebootConfig = _messages.EnumField('RebootConfigValueValuesEnum', 6)
    windowsUpdate = _messages.MessageField('WindowsUpdateSettings', 7)
    yum = _messages.MessageField('YumSettings', 8)
    zypper = _messages.MessageField('ZypperSettings', 9)