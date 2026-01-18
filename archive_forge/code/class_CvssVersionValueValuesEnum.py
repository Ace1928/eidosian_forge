from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CvssVersionValueValuesEnum(_messages.Enum):
    """Output only. CVSS version used to populate cvss_score and severity.

    Values:
      CVSS_VERSION_UNSPECIFIED: <no description>
      CVSS_VERSION_2: <no description>
      CVSS_VERSION_3: <no description>
    """
    CVSS_VERSION_UNSPECIFIED = 0
    CVSS_VERSION_2 = 1
    CVSS_VERSION_3 = 2