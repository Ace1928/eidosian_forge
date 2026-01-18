from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RadioTechnologyValueValuesEnum(_messages.Enum):
    """Conditional. This field specifies the radio access technology that is
    used for the CBSD.

    Values:
      RADIO_TECHNOLOGY_UNSPECIFIED: <no description>
      E_UTRA: <no description>
      CAMBIUM_NETWORKS: <no description>
      FOUR_G_BBW_SAA_1: <no description>
      NR: <no description>
      DOODLE_CBRS: <no description>
      CW: <no description>
      REDLINE: <no description>
      TARANA_WIRELESS: <no description>
    """
    RADIO_TECHNOLOGY_UNSPECIFIED = 0
    E_UTRA = 1
    CAMBIUM_NETWORKS = 2
    FOUR_G_BBW_SAA_1 = 3
    NR = 4
    DOODLE_CBRS = 5
    CW = 6
    REDLINE = 7
    TARANA_WIRELESS = 8