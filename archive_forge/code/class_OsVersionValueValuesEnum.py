from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsVersionValueValuesEnum(_messages.Enum):
    """OSVersion specifies the Windows node config to be used on the node

    Values:
      OS_VERSION_UNSPECIFIED: When OSVersion is not specified
      OS_VERSION_LTSC2019: LTSC2019 specifies to use LTSC2019 as the Windows
        Servercore Base Image
      OS_VERSION_LTSC2022: LTSC2022 specifies to use LTSC2022 as the Windows
        Servercore Base Image
    """
    OS_VERSION_UNSPECIFIED = 0
    OS_VERSION_LTSC2019 = 1
    OS_VERSION_LTSC2022 = 2