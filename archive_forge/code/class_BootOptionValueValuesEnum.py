from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BootOptionValueValuesEnum(_messages.Enum):
    """Output only. The VM Boot Option.

    Values:
      BOOT_OPTION_UNSPECIFIED: The boot option is unknown.
      EFI: The boot option is EFI.
      BIOS: The boot option is BIOS.
    """
    BOOT_OPTION_UNSPECIFIED = 0
    EFI = 1
    BIOS = 2